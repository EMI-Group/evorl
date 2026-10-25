from flax import nnx
import logging
import time

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.metrics import MetricBase, metric_field
from evorl.types import PyTreeDict, State
from evorl.algorithms.td3 import TD3TrainMetric

from evorl.algorithms.erl.erl_td3.erl_workflow import (
    ERLTrainMetric as ERLTrainMetricBase,
)
from evorl.algorithms.episodic_utils import create_dummy_td3_trainmetric
from evorl.algorithms.erl.erl_td3.erl_workflow import (
    make_erl_rollout_state,
    update_erl_td3,
)
from evorl.algorithms.erl.erl_td3.erl_ga import ERLGAWorkflow


logger = logging.getLogger(__name__)


class ERLTrainMetric(ERLTrainMetricBase):
    num_updates_per_iter: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )
    time_cost_per_iter: float = 0.0


class ERLOriginalWorkflow(ERLGAWorkflow):
    """Original ERL impl.

    Have dynamic training updates per iteration, i.e., #rl_updates = #sampled_timesteps_this_iter. Therefore the `step()` function cannot be directly jitted.
    """

    @classmethod
    def name(cls):
        return "ERL-Origin"

    def _ec_update(self, ec_opt_state, fitnesses):
        return self.ec_optimizer.tell(ec_opt_state, fitnesses)

    def _ec_update_with_rl_injection(self, ec_opt_state, agent_state, fitnesses):
        ec_opt_state = self._rl_injection(ec_opt_state, agent_state)
        return self.ec_optimizer.tell_external(ec_opt_state, fitnesses)

    def _rl_update(self, agent_state, opt_state, replay_buffer_state, key, num_updates):
        # The original algorithm uses a runtime training budget. Keep only the
        # last metrics in the carry instead of stacking every update's outputs.
        def sample_and_update(carry, unused_t):
            key, agent_state, opt_state, _ = carry
            key, replay_key, learn_key = jax.random.split(key, 3)
            sample_batches = jax.vmap(
                lambda sample_key: self.replay_buffer.sample(
                    replay_buffer_state, sample_key
                )
            )(
                jax.random.split(
                    replay_key,
                    self.config.actor_update_interval * self.config.num_rl_agents,
                )
            )
            sample_batches = jax.tree.map(
                lambda x: x.reshape(
                    (
                        self.config.actor_update_interval,
                        self.config.num_rl_agents,
                        *x.shape[1:],
                    )
                ),
                sample_batches,
            )
            (
                (agent_state, opt_state),
                (critic_loss, actor_loss, critic_losses, actor_losses),
            ) = update_erl_td3(
                agent_state,
                opt_state,
                sample_batches,
                learn_key,
                agent=self.agent,
                agent_state_vmap_axes=self.agent_state_vmap_axes,
                config=self.config,
            )
            metrics = TD3TrainMetric(
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                raw_loss_dict=PyTreeDict({**critic_losses, **actor_losses}),
            )
            return (key, agent_state, opt_state, metrics), None

        initial_metrics = create_dummy_td3_trainmetric(self.config.num_rl_agents)
        (_, agent_state, opt_state, metrics), _ = jax.lax.scan(
            sample_and_update,
            (key, agent_state, opt_state, initial_metrics),
            None,
            length=num_updates,
        )
        return metrics, agent_state, opt_state

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
        pop_agent_state = make_erl_rollout_state(agent_state, pop_actor_params)
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
    if isinstance(pop, nnx.State):
        pop = nnx.to_pure_dict(pop)

    def _get_stats(x):
        return dict(
            min=jnp.min(x).tolist(),
            max=jnp.max(x).tolist(),
        )

    return jtu.tree_map(_get_stats, pop)
