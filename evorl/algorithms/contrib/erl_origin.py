import logging
import math
import time
from functools import partial
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import orbax.checkpoint as ocp

from evorl.metrics import MetricBase, metricfield
from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import is_jitted
from evorl.recorders import get_1d_array_statistics, add_prefix

from ..td3 import TD3TrainMetric
from ..erl.erl_ga import ERLGAWorkflow, replace_td3_actor_params
from ..offpolicy_utils import skip_replay_buffer_state


logger = logging.getLogger(__name__)


class POPTrainMetric(MetricBase):
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array
    num_updates_per_iter: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    rb_size: int = 0
    rl_episode_returns: chex.Array | None = None
    rl_episode_lengths: chex.Array | None = None
    rl_metrics: MetricBase | None = None
    time_cost_per_iter: float = 0.0
    ec_info: PyTreeDict = metricfield(default_factory=PyTreeDict)


class ERLWorkflow(ERLGAWorkflow):
    """
    Original ERL impl with dynamic training updates per iteration, i.e., #rl_updates = #sampled_timesteps_this_iter
    """

    @classmethod
    def name(cls):
        return "ERL-Origin"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        """
        return workflow
        """

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
            sample_batches = jax.tree_map(
                lambda x: x.reshape(
                    (
                        config.actor_update_interval,
                        config.num_rl_agents,
                        *x.shape[1:],
                    )
                ),
                sample_batches,
            )

            (agent_state, opt_state), train_info = workflow._rl_update_fn(
                agent_state, opt_state, sample_batches, learn_key
            )

            # Note: we do not put train_info into y_t for saving memory
            return (key, agent_state, opt_state, replay_buffer_state, train_info), None

        if is_jitted(cls.evaluate):
            _rl_sample_and_update_fn = jax.jit(_rl_sample_and_update_fn)

        workflow._rl_sample_and_update_fn = _rl_sample_and_update_fn

        return workflow

    def _ec_update(self, ec_opt_state, fitnesses):
        return self.ec_optimizer.tell(ec_opt_state, fitnesses)

    def _rl_update(self, agent_state, opt_state, replay_buffer_state, key, num_updates):
        # unlike erl-ga, since num_updates is large, we only use the last train_info
        init_train_info = (
            jnp.zeros(()),
            jnp.zeros(()),
            PyTreeDict(
                critic_loss=jnp.zeros((self.config.num_rl_agents,)),
                q_value=jnp.zeros((self.config.num_rl_agents,)),
            ),
            PyTreeDict(actor_loss=jnp.zeros((self.config.num_rl_agents,))),
        )

        (_, agent_state, opt_state, replay_buffer_state, train_info), _ = jax.lax.scan(
            self._rl_sample_and_update_fn,
            (key, agent_state, opt_state, replay_buffer_state, init_train_info),
            (),
            length=num_updates,
        )

        critic_loss, actor_loss, critic_loss_dict, actor_loss_dict = train_info

        # smoothed td3 metrics
        td3_metrics = TD3TrainMetric(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
        )

        return td3_metrics, agent_state, opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        """
        the basic step function for the workflow to update agent
        """
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

        # ======== EC update ========
        # the trajectory [#pop, T, B, ...]
        # metrics: [#pop, B]
        pop_actor_params, ec_opt_state = self.ec_optimizer.ask(ec_opt_state)
        pop_agent_state = replace_td3_actor_params(agent_state, pop_actor_params)
        ec_eval_metrics, ec_trajectory, replay_buffer_state = self._ec_rollout(
            pop_agent_state, replay_buffer_state, ec_rollout_key
        )

        fitnesses = ec_eval_metrics.episode_returns.mean(axis=-1)
        ec_metrics, ec_opt_state = self._ec_update(ec_opt_state, fitnesses)

        # calculate the number of timestep
        sampled_timesteps += ec_eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        sampled_episodes += jnp.uint32(self.config.episodes_for_fitness * pop_size)

        train_metrics = POPTrainMetric(
            pop_episode_lengths=ec_eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=ec_eval_metrics.episode_returns.mean(-1),
            ec_info=ec_metrics,
        )

        # ======== RL update ========
        if iterations > self.config.warmup_iters:
            rl_eval_metrics, rl_trajectory, replay_buffer_state = self._rl_rollout(
                agent_state, replay_buffer_state, rl_rollout_key
            )

            rl_sampled_timesteps = rl_eval_metrics.episode_lengths.sum().astype(
                jnp.uint32
            )
            sampled_timesteps += rl_sampled_timesteps
            sampled_episodes += jnp.uint32(
                self.config.num_rl_agents * self.config.rollout_episodes
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
                raise ValueError(
                    f"Unknown rl_updates_mode: {self.config.rl_updates_mode}"
                )

            td3_metrics, agent_state, opt_state = self._rl_update(
                agent_state, opt_state, replay_buffer_state, learn_key, num_updates
            )

            # get average loss
            td3_metrics = td3_metrics.replace(
                actor_loss=td3_metrics.actor_loss / self.config.num_rl_agents,
                critic_loss=td3_metrics.critic_loss / self.config.num_rl_agents,
            )

            if iterations % self.config.rl_injection_interval == 0:
                ec_opt_state = self._rl_injection(ec_opt_state, agent_state, fitnesses)

            train_metrics = train_metrics.replace(
                num_updates_per_iter=num_updates,
                rl_episode_lengths=rl_eval_metrics.episode_lengths.mean(-1),
                rl_episode_returns=rl_eval_metrics.episode_returns.mean(-1),
                rl_metrics=td3_metrics,
            )

        else:
            rl_sampled_timesteps = jnp.zeros((), dtype=jnp.uint32)

        train_metrics = train_metrics.replace(
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

    def learn(self, state: State) -> State:
        num_iters = math.ceil(
            self.config.total_episodes
            / (self.config.episodes_for_fitness * self.config.pop_size)
        )

        for i in range(state.metrics.iterations, num_iters):
            iters = i + 1

            train_metrics, state = self.step(state)

            workflow_metrics = state.metrics

            workflow_metrics_dict = workflow_metrics.to_local_dict()
            self.recorder.write(workflow_metrics_dict, iters)

            train_metrics_dict = train_metrics.to_local_dict()
            train_metrics_dict["pop_episode_returns"] = get_1d_array_statistics(
                train_metrics_dict["pop_episode_returns"], histogram=True
            )

            train_metrics_dict["pop_episode_lengths"] = get_1d_array_statistics(
                train_metrics_dict["pop_episode_lengths"], histogram=True
            )

            if train_metrics_dict["rl_metrics"] is not None:
                if self.config.num_rl_agents > 1:
                    train_metrics_dict["rl_episode_lengths"] = get_1d_array_statistics(
                        train_metrics_dict["rl_episode_lengths"], histogram=True
                    )
                    train_metrics_dict["rl_episode_returns"] = get_1d_array_statistics(
                        train_metrics_dict["rl_episode_returns"], histogram=True
                    )
                    train_metrics_dict["rl_metrics"]["raw_loss_dict"] = jtu.tree_map(
                        get_1d_array_statistics,
                        train_metrics_dict["rl_metrics"]["raw_loss_dict"],
                    )
                else:
                    train_metrics_dict["rl_episode_lengths"] = train_metrics_dict[
                        "rl_episode_lengths"
                    ].squeeze(0)
                    train_metrics_dict["rl_episode_returns"] = train_metrics_dict[
                        "rl_episode_returns"
                    ].squeeze(0)

            self.recorder.write(train_metrics_dict, iters)

            pop_statistics = get_ec_pop_statistics(state.ec_opt_state.pop)
            self.recorder.write(add_prefix(pop_statistics, "ec"), iters)

            if iters % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)

                eval_metrics_dict = eval_metrics.to_local_dict()
                if self.config.num_rl_agents > 1:
                    eval_metrics_dict = jtu.tree_map(
                        partial(get_1d_array_statistics, histogram=True),
                        eval_metrics_dict,
                    )

                self.recorder.write(add_prefix(eval_metrics_dict, "eval"), iters)

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)

            self.checkpoint_manager.save(iters, args=ocp.args.StandardSave(state))

        return state

    @classmethod
    def enable_jit(cls) -> None:
        """
        Do not jit replay buffer add
        """
        cls._rl_rollout = jax.jit(cls._rl_rollout, static_argnums=(0,))
        cls._ec_rollout = jax.jit(cls._ec_rollout, static_argnums=(0,))
        cls._ec_update = jax.jit(cls._ec_update, static_argnums=(0,))
        cls._rl_injection = jax.jit(cls._rl_injection, static_argnums=(0,))

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
