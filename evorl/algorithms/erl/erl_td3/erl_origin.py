import logging
import time
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.metrics import MetricBase
from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import is_jitted
from evorl.algorithms.td3 import TD3TrainMetric

from ..erl_workflow import ERLTrainMetric as ERLTrainMetricBase
from .erl_td3_workflow import create_dummy_td3_trainmetric, erl_replace_td3_actor_params
from .erl_ga import ERLGAWorkflow


logger = logging.getLogger(__name__)


class ERLTrainMetric(ERLTrainMetricBase):
    num_updates_per_iter: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    time_cost_per_iter: float = 0.0


class ERLWorkflow(ERLGAWorkflow):
    """Original ERL impl.

    Have dynamic training updates per iteration, i.e., #rl_updates = #sampled_timesteps_this_iter. Therefore the `step()` function cannot be directly jitted.
    """

    @classmethod
    def name(cls):
        return "ERL-Origin"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        workflow = super()._build_from_config(config)

        def _rl_sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state, replay_buffer_state, _ = carry

            def _sample_fn(key):
                return workflow.replay_buffer.sample(replay_buffer_state, key)

            key, rb_key, learn_key = jax.random.split(key, 3)

            rb_keys = jax.random.split(
                rb_key, config.actor_update_interval * config.num_rl_agents
            )
            sample_batches = jax.vmap(_sample_fn)(rb_keys)

            # (actor_update_interval, num_learning_offspring, B, ...)
            sample_batches = jtu.tree_map(
                lambda x: x.reshape(
                    (
                        config.actor_update_interval,
                        config.num_rl_agents,
                        *x.shape[1:],
                    )
                ),
                sample_batches,
            )

            (
                (agent_state, opt_state),
                (
                    critic_loss,
                    actor_loss,
                    critic_loss_dict,
                    actor_loss_dict,
                ),
            ) = workflow._rl_update_fn(
                agent_state, opt_state, sample_batches, learn_key
            )

            td3_metrics = TD3TrainMetric(
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
            )

            # Note: we do not put train_info into y_t for saving memory
            return (key, agent_state, opt_state, replay_buffer_state, td3_metrics), None

        if is_jitted(cls.evaluate):
            _rl_sample_and_update_fn = jax.jit(_rl_sample_and_update_fn)

        workflow._rl_sample_and_update_fn = _rl_sample_and_update_fn

        return workflow

    def _ec_update(self, ec_opt_state, fitnesses):
        return self.ec_optimizer.tell(ec_opt_state, fitnesses)

    def _ec_update_with_rl_injection(self, ec_opt_state, agent_state, fitnesses):
        ec_opt_state = self._rl_injection(ec_opt_state, agent_state)
        return self.ec_optimizer.tell_external(ec_opt_state, fitnesses)

    def _rl_update(self, agent_state, opt_state, replay_buffer_state, key, num_updates):
        # unlike erl-ga, since num_updates is large, we only use the last train_info
        init_td3_metrics = create_dummy_td3_trainmetric(self.config.num_rl_agents)

        (_, agent_state, opt_state, replay_buffer_state, td3_metrics), _ = jax.lax.scan(
            self._rl_sample_and_update_fn,
            (key, agent_state, opt_state, replay_buffer_state, init_td3_metrics),
            (),
            length=num_updates,
        )

        return td3_metrics, agent_state, opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        """The basic step function for the workflow to update agent."""
        start_t = time.perf_counter()
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        opt_state = state.opt_state
        ec_opt_state = state.ec_opt_state
        replay_buffer_state = state.replay_buffer_state
        iterations = state.metrics.iterations + 1

        sampled_timesteps = jnp.zeros((), dtype=jnp.uint32)
        sampled_episodes = jnp.zeros((), dtype=jnp.uint32)

        key, ec_rollout_key, rl_rollout_key, learn_key = jax.random.split(
            state.key, num=4
        )

        # ======== EC rollout ========
        # the trajectory [#pop, T, B, ...]
        # metrics: [#pop, B]
        pop_actor_params, ec_opt_state = self.ec_optimizer.ask(ec_opt_state)
        pop_agent_state = erl_replace_td3_actor_params(agent_state, pop_actor_params)
        ec_eval_metrics, ec_trajectory, replay_buffer_state = self._ec_rollout(
            pop_agent_state, replay_buffer_state, ec_rollout_key
        )

        # calculate the number of timestep
        sampled_timesteps += ec_eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        sampled_episodes += jnp.uint32(self.config.episodes_for_fitness * pop_size)

        train_metrics = ERLTrainMetric(
            pop_episode_lengths=ec_eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=ec_eval_metrics.episode_returns.mean(-1),
        )

        # ======== RL update ========

        rl_eval_metrics, rl_trajectory, replay_buffer_state = self._rl_rollout(
            agent_state, replay_buffer_state, rl_rollout_key
        )

        if self.config.rl_updates_mode == "global":  # same as original ERL
            total_timesteps = state.metrics.sampled_timesteps + sampled_timesteps
            num_updates = (
                jnp.ceil(total_timesteps * self.config.rl_updates_frac).astype(
                    jnp.uint32
                )
                // self.config.actor_update_interval
            )
        elif self.config.rl_updates_mode == "iter":
            num_updates = (
                jnp.ceil(sampled_timesteps * self.config.rl_updates_frac).astype(
                    jnp.uint32
                )
                // self.config.actor_update_interval
            )
        else:
            raise ValueError(f"Unknown rl_updates_mode: {self.config.rl_updates_mode}")

        td3_metrics, agent_state, opt_state = self._rl_update(
            agent_state, opt_state, replay_buffer_state, learn_key, num_updates
        )

        # get average loss
        td3_metrics = td3_metrics.replace(
            actor_loss=td3_metrics.actor_loss / self.config.num_rl_agents,
            critic_loss=td3_metrics.critic_loss / self.config.num_rl_agents,
        )

        train_metrics = train_metrics.replace(
            num_updates_per_iter=num_updates,
            rl_episode_lengths=rl_eval_metrics.episode_lengths.mean(-1),
            rl_episode_returns=rl_eval_metrics.episode_returns.mean(-1),
            rl_metrics=td3_metrics,
        )

        rl_sampled_timesteps = rl_eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        sampled_timesteps += rl_sampled_timesteps
        sampled_episodes += jnp.uint32(
            self.config.num_rl_agents * self.config.rollout_episodes
        )

        # ======== EC update ========
        fitnesses = ec_eval_metrics.episode_returns.mean(axis=-1)

        if iterations % self.config.rl_injection_interval == 0:
            ec_metrics, ec_opt_state = self._ec_update_with_rl_injection(
                ec_opt_state, agent_state, fitnesses
            )
        else:
            ec_metrics, ec_opt_state = self._ec_update(ec_opt_state, fitnesses)

        train_metrics = train_metrics.replace(
            ec_info=ec_metrics,
            rb_size=replay_buffer_state.buffer_size,
            time_cost_per_iter=time.perf_counter() - start_t,
        )

        # iterations is the number of updates of the agent
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
            rl_sampled_timesteps=state.metrics.rl_sampled_timesteps
            + rl_sampled_timesteps,
            iterations=iterations,
        )

        state = state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
            ec_opt_state=ec_opt_state,
            opt_state=opt_state,
        )

        return train_metrics, state

    @classmethod
    def enable_jit(cls) -> None:
        # Do not jit replay buffer add

        cls._rl_rollout = jax.jit(cls._rl_rollout, static_argnums=(0,))
        cls._ec_rollout = jax.jit(cls._ec_rollout, static_argnums=(0,))
        cls._ec_update = jax.jit(cls._ec_update, static_argnums=(0,))
        cls._ec_update_with_rl_injection = jax.jit(
            cls._ec_update_with_rl_injection, static_argnums=(0,)
        )

        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )


def get_ec_pop_statistics(pop):
    pop = pop["params"]

    def _get_stats(x):
        return dict(
            min=jnp.min(x).tolist(),
            max=jnp.max(x).tolist(),
        )

    return jtu.tree_map(_get_stats, pop)
