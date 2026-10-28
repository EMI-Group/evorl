from flax import nnx
import time

import jax
import jax.numpy as jnp

from evorl.metrics import MetricBase, metric_field
from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import (
    tree_get,
    tree_set,
)
from evorl.algorithms.erl.cemrl_td3.cemrl import CEMRLWorkflow
from evorl.algorithms.td3 import TD3TrainMetric
from evorl.algorithms.episodic_utils import create_dummy_td3_trainmetric

from evorl.algorithms.erl.cemrl_td3.cemrl_workflow import (
    CEMRLTrainMetric as CEMRLTrainMetricBase,
    replace_cemrl_actor_params,
    update_cemrl_td3,
)


class CEMRLTrainMetric(CEMRLTrainMetricBase):
    num_updates_per_iter: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )
    time_cost_per_iter: float = 0.0


class WorkflowMetric(MetricBase):
    sampled_timesteps: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )
    sampled_timesteps_per_iter: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )
    sampled_episodes: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )
    iterations: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )


class CEMRLOriginalWorkflow(CEMRLWorkflow):
    """Original CEMRL impl.

    1 critic + n actors + 1 replay buffer.
    """

    @classmethod
    def name(cls):
        return "CEMRL-Origin"

    def _setup_workflow_metrics(self) -> MetricBase:
        return WorkflowMetric()

    def _ec_sample(self, ec_opt_state):
        return self.ec_optimizer.ask(ec_opt_state)

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
                    self.config.actor_update_interval
                    * self.config.num_learning_offspring,
                )
            )
            sample_batches = jax.tree.map(
                lambda x: x.reshape(
                    (
                        self.config.actor_update_interval,
                        self.config.num_learning_offspring,
                        *x.shape[1:],
                    )
                ),
                sample_batches,
            )
            (
                (agent_state, opt_state),
                (critic_loss, actor_loss, critic_losses, actor_losses),
            ) = update_cemrl_td3(
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

        initial_metrics = create_dummy_td3_trainmetric(
            self.config.num_learning_offspring
        )
        (_, agent_state, opt_state, metrics), _ = jax.lax.scan(
            sample_and_update,
            (key, agent_state, opt_state, initial_metrics),
            None,
            length=num_updates,
        )
        metrics = metrics.replace(
            actor_loss=metrics.actor_loss / self.config.num_learning_offspring
        )
        return metrics, agent_state, opt_state

    def _rollout_and_update(
        self, pop_agent_state, replay_buffer_state, ec_opt_state, key
    ):
        """Calculate the fitness and update the replay buffer and ec_optimizer."""
        # the trajectory [T, #pop*B, ...]
        # metrics: [#pop, B]
        eval_metrics, trajectory, replay_buffer_state = self._rollout(
            pop_agent_state, replay_buffer_state, key
        )

        fitnesses = eval_metrics.episode_return.mean(axis=-1)
        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        return eval_metrics, ec_metrics, fitnesses, replay_buffer_state, ec_opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        start_t = time.perf_counter()
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        opt_state = state.opt_state
        ec_opt_state = state.ec_opt_state
        replay_buffer_state = state.replay_buffer_state
        iterations = state.metrics.iterations + 1

        key, perm_key, rollout_key, learn_key = jax.random.split(state.key, num=4)

        # ======= CEM Sample ========
        pop_actor_params, ec_opt_state = self._ec_sample(ec_opt_state)

        # ======== RL update ========

        if iterations > self.config.warmup_iters:
            learning_actor_indices = jax.random.choice(
                perm_key,
                self.config.pop_size,
                (self.config.num_learning_offspring,),
                replace=False,
            )
            learning_actor_params = tree_get(pop_actor_params, learning_actor_indices)
            learning_agent_state = replace_cemrl_actor_params(
                agent_state, learning_actor_params
            )
            # reset and add actors' opt_state
            learning_opt_state = opt_state.replace(
                actor=nnx.unpack(
                    nnx.vmap(
                        lambda model: nnx.Optimizer(
                            model, self.optimizer, wrt=nnx.Param
                        )
                    )(nnx.merge(learning_agent_state.model_state, copy=True).actor)
                ),
            )

            num_updates = (
                jnp.ceil(
                    state.metrics.sampled_timesteps_per_iter
                    * self.config.rl_updates_frac
                ).astype(jnp.uint32)
                // self.config.actor_update_interval
            )

            td3_metrics, learning_agent_state, learning_opt_state = self._rl_update(
                learning_agent_state,
                learning_opt_state,
                replay_buffer_state,
                learn_key,
                num_updates,
            )

            pop_actor_params = tree_set(
                pop_actor_params,
                nnx.state(
                    nnx.merge(learning_agent_state.model_state, copy=True).actor,
                    nnx.Param,
                ),
                learning_actor_indices,
                unique_indices=True,
            )

            # drop the actors and their opt_state
            agent_state = replace_cemrl_actor_params(
                learning_agent_state, actor_params=None
            )
            opt_state = learning_opt_state.replace(actor=None)

            # rl injection
            ec_opt_state = self._rl_injection(ec_opt_state, pop_actor_params)

        else:
            num_updates = jnp.zeros((), dtype=jnp.uint32)
            td3_metrics = None

        # ======== CEM update ========
        pop_agent_state = replace_cemrl_actor_params(agent_state, pop_actor_params)
        eval_metrics, ec_metrics, fitnesses, replay_buffer_state, ec_opt_state = (
            self._rollout_and_update(
                pop_agent_state,
                replay_buffer_state,
                ec_opt_state,
                rollout_key,
            )
        )

        # adding debug info for CEM
        ec_info = PyTreeDict(ec_metrics)
        ec_info.cov_eps = ec_opt_state.cov_eps
        if td3_metrics is not None:
            elites_indices = jax.lax.top_k(fitnesses, self.config.num_elites)[1]
            elites_from_rl = jnp.isin(learning_actor_indices, elites_indices)
            ec_info.elites_from_rl = elites_from_rl.sum()
            ec_info.elites_from_rl_ratio = elites_from_rl.mean()

        train_metrics = CEMRLTrainMetric(
            rb_size=replay_buffer_state.buffer_size,
            pop_episode_length=eval_metrics.episode_length.mean(-1),
            pop_episode_return=eval_metrics.episode_return.mean(-1),
            rl_metrics=td3_metrics,
            ec_info=ec_info,
            num_updates_per_iter=num_updates,
            time_cost_per_iter=time.perf_counter() - start_t,
        )

        # calculate the number of timestep
        sampled_timesteps = eval_metrics.episode_length.sum().astype(jnp.uint32)
        sampled_episodes = jnp.uint32(self.config.episodes_for_fitness * pop_size)

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
