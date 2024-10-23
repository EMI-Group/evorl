import time
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp

from evorl.metrics import MetricBase, metric_field
from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import (
    tree_get,
    tree_set,
    scan_and_last,
    is_jitted,
)

from ..td3 import TD3TrainMetric
from ..erl.cemrl import (
    replace_actor_params,
    CEMRLWorkflow as _CEMRLWorkflow,
)


class POPTrainMetric(MetricBase):
    rb_size: int
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array
    rl_metrics: MetricBase | None = None
    num_updates_per_iter: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    time_cost_per_iter: float = 0.0
    ec_info: PyTreeDict = metric_field(default_factory=PyTreeDict)


class WorkflowMetric(MetricBase):
    sampled_timesteps: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_timesteps_per_iter: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class CEMRLWorkflow(_CEMRLWorkflow):
    """
    1 critic + n actors + 1 replay buffer.
    We use shard_map to split and parallel the population.
    """

    @classmethod
    def name(cls):
        return "CEMRL-Origin"

    def _setup_workflow_metrics(self) -> MetricBase:
        return WorkflowMetric()

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        workflow = super()._build_from_config(config)

        def _rl_sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state, replay_buffer_state = carry

            def _sample_fn(key):
                return workflow.replay_buffer.sample(replay_buffer_state, key)

            key, rb_key, learn_key = jax.random.split(key, 3)
            rb_keys = jax.random.split(
                rb_key,
                config.actor_update_interval * config.num_learning_offspring,
            )
            sample_batches = jax.vmap(_sample_fn)(rb_keys)

            # (actor_update_interval, num_learning_offspring, B, ...)
            sample_batches = jax.tree_map(
                lambda x: x.reshape(
                    (
                        config.actor_update_interval,
                        config.num_learning_offspring,
                        *x.shape[1:],
                    )
                ),
                sample_batches,
            )

            (agent_state, opt_state), train_info = workflow._rl_update_fn(
                agent_state, opt_state, sample_batches, learn_key
            )

            return (key, agent_state, opt_state, replay_buffer_state), train_info

        if is_jitted(cls.evaluate):
            _rl_sample_and_update_fn = jax.jit(_rl_sample_and_update_fn)

        workflow._rl_sample_and_update_fn = _rl_sample_and_update_fn

        return workflow

    def _ec_sample(self, ec_opt_state):
        return self.ec_optimizer.ask(ec_opt_state)

    def _rl_update(
        self,
        agent_state,
        opt_state,
        replay_buffer_state,
        pop_actor_params,
        key,
        num_updates,
    ):
        """
        Add num_updates support. Therefore this method cannot be jitted.
        """
        if self.config.mirror_sampling:
            key, perm_key = jax.random.split(key)
            learning_actor_indices = jax.random.choice(
                perm_key,
                self.config.pop_size,
                (self.config.num_learning_offspring,),
                replace=False,
            )
        else:
            learning_actor_indices = slice(
                self.config.num_learning_offspring
            )  # [:self.config.num_learning_offspring]
        learning_actor_params = tree_get(pop_actor_params, learning_actor_indices)
        learning_agent_state = replace_actor_params(agent_state, learning_actor_params)

        # reset and add actors' opt_state
        learning_opt_state = opt_state.replace(
            actor=self.optimizer.init(learning_actor_params),
        )

        (
            (_, learning_agent_state, learning_opt_state, replay_buffer_state),
            train_info,
        ) = scan_and_last(
            self._rl_sample_and_update_fn,
            (key, learning_agent_state, learning_opt_state, replay_buffer_state),
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

        pop_actor_params = tree_set(
            pop_actor_params,
            learning_agent_state.params.actor_params,
            learning_actor_indices,
            unique_indices=True,
        )

        # drop the actors and their opt_state
        agent_state = replace_actor_params(learning_agent_state, pop_actor_params=None)
        opt_state = learning_opt_state.replace(actor=None)

        td3_metrics = td3_metrics.replace(
            actor_loss=td3_metrics.actor_loss / self.config.num_learning_offspring
        )

        return td3_metrics, pop_actor_params, agent_state, opt_state

    def _rollout_and_update(
        self, agent_state, replay_buffer_state, ec_opt_state, pop_actor_params, key
    ):
        """
        Calculate the fitness and update the replay buffer and ec_optimizer
        """
        # Note: updated critic_params are stored in learning_agent_state
        # actor_params [num_learning_offspring, ...] -> [pop_size, ...]
        # reset target_actor_params
        pop_agent_state = replace_actor_params(agent_state, pop_actor_params)

        # the trajectory [T, #pop*B, ...]
        # metrics: [#pop, B]
        eval_metrics, trajectory, replay_buffer_state = self._rollout(
            pop_agent_state, replay_buffer_state, key
        )

        fitnesses = eval_metrics.episode_returns.mean(axis=-1)
        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        return eval_metrics, ec_metrics, fitnesses, replay_buffer_state, ec_opt_state

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

        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        # ======= CEM Sample ========
        pop_actor_params, ec_opt_state = self._ec_sample(ec_opt_state)

        # ======== RL update ========
        if iterations > self.config.warmup_iters:
            num_updates = (
                jnp.ceil(
                    state.metrics.sampled_timesteps_per_iter
                    * self.config.rl_updates_frac
                ).astype(jnp.uint32)
                // self.config.actor_update_interval
            )

            td3_metrics, pop_actor_params, agent_state, opt_state = self._rl_update(
                agent_state,
                opt_state,
                replay_buffer_state,
                pop_actor_params,
                learn_key,
                num_updates,
            )

        else:
            num_updates = jnp.zeros((), dtype=jnp.uint32)
            td3_metrics = None

        # ======== CEM update ========

        eval_metrics, ec_metrics, fitnesses, replay_buffer_state, ec_opt_state = (
            self._rollout_and_update(
                agent_state,
                replay_buffer_state,
                ec_opt_state,
                pop_actor_params,
                rollout_key,
            )
        )

        train_metrics = POPTrainMetric(
            rb_size=replay_buffer_state.buffer_size,
            pop_episode_lengths=eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=eval_metrics.episode_returns.mean(-1),
            rl_metrics=td3_metrics,
            num_updates_per_iter=num_updates,
            time_cost_per_iter=time.perf_counter() - start_t,
        )

        # adding debug info for CEM
        ec_info = PyTreeDict(ec_metrics)
        ec_info.cov_eps = ec_opt_state.cov_eps
        if td3_metrics is not None:
            elites_indices = jax.lax.top_k(fitnesses, self.config.num_elites)[1]
            elites_from_rl = jnp.isin(
                jnp.arange(self.config.num_learning_offspring), elites_indices
            )
            ec_info.elites_from_rl = elites_from_rl.sum()
            ec_info.elites_from_rl_ratio = elites_from_rl.mean()

        train_metrics = train_metrics.replace(ec_info=ec_info)

        # calculate the number of timestep
        sampled_timesteps = eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        sampled_episodes = jnp.uint32(self.config.episodes_for_fitness * pop_size)

        # iterations is the number of updates of the agent

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_timesteps_per_iter=sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
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
        cls._ec_sample = jax.jit(cls._ec_sample, static_argnums=(0,))
        cls._rollout_and_update = jax.jit(cls._rollout_and_update, static_argnums=(0,))

        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )
