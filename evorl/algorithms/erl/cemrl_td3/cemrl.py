from flax import nnx
import math
import numpy as np
from omegaconf import DictConfig

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.metrics import MetricBase
from evorl.types import PyTreeDict, State, Params
from evorl.utils.jax_utils import tree_get, tree_set
from evorl.recorders import get_1d_array_statistics, add_prefix
from evorl.ec.optimizers import SepCEM, ECState, ExponentialScheduleSpec
from evorl.algorithms.offpolicy_utils import skip_replay_buffer_state

import logging
from evorl.algorithms.episodic_utils import (
    create_dummy_td3_trainmetric,
)

from .cemrl_workflow import (
    CEMRLWorkflowBase,
    CEMRLTrainMetric,
    replace_cemrl_actor_params,
)

logger = logging.getLogger(__name__)


class CEMRLWorkflow(CEMRLWorkflowBase):
    """A CEM actor population with one shared TD3 critic and replay buffer."""

    @classmethod
    def name(cls):
        return "CEMRL"

    @classmethod
    def _build_ec_optimizer(cls, config: DictConfig):
        return SepCEM(
            pop_size=config.pop_size,
            num_elites=config.num_elites,
            cov_eps_schedule=ExponentialScheduleSpec(**config.cov_eps),
            weighted_update=config.weighted_update,
            rank_weight_shift=config.rank_weight_shift,
            mirror_sampling=config.mirror_sampling,
        )

    def _rl_injection(self, ec_opt_state: ECState, pop: Params) -> ECState:
        return ec_opt_state.replace(pop=pop)

    def step(self, state: State) -> tuple[MetricBase, State]:
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        opt_state = state.opt_state
        ec_opt_state = state.ec_opt_state
        replay_buffer_state = state.replay_buffer_state
        iterations = state.metrics.iterations + 1

        key, perm_key, rollout_key, learn_key = jax.random.split(state.key, num=4)

        # ======= CEM Sample ========
        pop_actor_params, ec_opt_state = self.ec_optimizer.ask(ec_opt_state)

        # ======== RL update ========
        learning_actor_indices = jax.random.choice(
            perm_key,
            self.config.pop_size,
            (self.config.num_learning_offspring,),
            replace=False,
        )

        def train_offspring(agent_state, opt_state, pop_actor_params):
            learning_actor_params = tree_get(pop_actor_params, learning_actor_indices)
            learning_agent_state = replace_cemrl_actor_params(
                agent_state, learning_actor_params
            )

            # reset actors' opt_state
            learning_opt_state = opt_state.replace(
                actor=nnx.unpack(
                    nnx.vmap(
                        lambda model: nnx.Optimizer(
                            model, self.optimizer, wrt=nnx.Param
                        )
                    )(nnx.merge(learning_agent_state.model_state, copy=True).actor)
                ),
            )

            td3_metrics, learning_agent_state, learning_opt_state = self._rl_update(
                learning_agent_state,
                learning_opt_state,
                replay_buffer_state,
                learn_key,
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
            return td3_metrics, pop_actor_params, agent_state, opt_state

        def skip_training(agent_state, opt_state, pop_actor_params):
            return (
                create_dummy_td3_trainmetric(self.config.num_learning_offspring),
                pop_actor_params,
                agent_state,
                opt_state,
            )

        td3_metrics, pop_actor_params, agent_state, opt_state = jax.lax.cond(
            iterations > self.config.warmup_iters,
            train_offspring,
            skip_training,
            agent_state,
            opt_state,
            pop_actor_params,
        )

        # ======== CEM update ========
        pop_agent_state = replace_cemrl_actor_params(agent_state, pop_actor_params)

        # the trajectory [T, #pop*B, ...]
        # metrics: [#pop, B]
        eval_metrics, trajectory, replay_buffer_state = self._rollout(
            pop_agent_state, replay_buffer_state, rollout_key
        )

        fitnesses = eval_metrics.episode_returns.mean(axis=-1)

        ec_opt_state = self._rl_injection(ec_opt_state, pop_actor_params)
        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        # adding debug info for CEM
        ec_info = PyTreeDict(ec_metrics)
        ec_info.cov_eps = ec_opt_state.cov_eps

        def _calc_elites_stats(ec_info):
            elites_indices = jax.lax.top_k(fitnesses, self.config.num_elites)[1]
            elites_from_rl = jnp.isin(learning_actor_indices, elites_indices).astype(
                jnp.int32
            )
            ec_info.elites_from_rl = elites_from_rl.sum()
            ec_info.elites_from_rl_ratio = elites_from_rl.mean()
            return ec_info

        def _dummy_calc_elites_stats(ec_info):
            ec_info.elites_from_rl = jnp.zeros((), dtype=jnp.int32)
            ec_info.elites_from_rl_ratio = jnp.zeros(())
            return ec_info

        ec_info = jax.lax.cond(
            iterations > self.config.warmup_iters,
            _calc_elites_stats,
            _dummy_calc_elites_stats,
            ec_info,
        )

        train_metrics = CEMRLTrainMetric(
            rb_size=replay_buffer_state.buffer_size,
            pop_episode_lengths=eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=eval_metrics.episode_returns.mean(-1),
            rl_metrics=td3_metrics,
            ec_info=ec_info,
        )

        # calculate the number of timestep
        sampled_timesteps = eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        sampled_episodes = jnp.uint32(self.config.episodes_for_fitness * pop_size)

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
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

    def learn(self, state: State) -> State:
        sampled_episodes = int(state.metrics.sampled_episodes)
        start_iteration = int(state.metrics.iterations)
        num_iters = math.ceil(
            (self.config.total_episodes - sampled_episodes)
            / (self.config.episodes_for_fitness * self.config.pop_size)
        )

        final_iteration = start_iteration + num_iters
        for i in range(start_iteration, final_iteration):
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
                train_metrics_dict["rl_metrics"]["raw_loss_dict"] = jtu.tree_map(
                    get_1d_array_statistics,
                    train_metrics_dict["rl_metrics"]["raw_loss_dict"],
                )

            self.recorder.write(train_metrics_dict, iters)

            std_statistics = get_std_statistics(state.ec_opt_state.variance)
            self.recorder.write({"ec/std": std_statistics}, iters)

            if iters % self.config.eval_interval == 0 or iters == final_iteration:
                eval_metrics, state = self.evaluate(state)

                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iters
                )

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)

            self.checkpoint_manager.save(
                iters,
                saved_state,
                force=iters == final_iteration,
            )

        return state


def get_std_statistics(variance):
    if isinstance(variance, nnx.State):
        variance = nnx.to_pure_dict(variance)

    def _get_stats(x):
        x = np.sqrt(x)
        return dict(
            min=np.min(x).tolist(),
            max=np.max(x).tolist(),
            mean=np.mean(x).tolist(),
        )

    return jtu.tree_map(_get_stats, variance)
