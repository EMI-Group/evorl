import logging
import math

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import orbax.checkpoint as ocp

from evorl.metrics import MetricBase
from evorl.types import State, PyTreeDict
from evorl.utils.flashbax_utils import get_buffer_size
from evorl.recorders import get_1d_array_statistics, add_prefix

from ..offpolicy_utils import skip_replay_buffer_state
from ..erl.cemrl import (
    CEMRLWorkflow,
    POPTrainMetric,
    replace_actor_params,
    get_std_statistics,
)


logger = logging.getLogger(__name__)


class EvaluateMetric(MetricBase):
    pop_center_episode_returns: chex.Array
    pop_center_episode_lengths: chex.Array
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array


class PopRLWorkflow(CEMRLWorkflow):
    @classmethod
    def name(cls):
        return "CEM-RL(nocem)"

    def step(self, state: State) -> tuple[MetricBase, State]:
        """
        the basic step function for the workflow to update agent
        """
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        opt_state = state.opt_state
        ec_opt_state = state.ec_opt_state
        replay_buffer_state = state.replay_buffer_state
        iterations = state.metrics.iterations + 1

        pop_actor_params = agent_state.params.actor_params

        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        # ======== RL update ========
        if iterations > self.config.warmup_iters:
            # reset and add actors' opt_state
            new_opt_state = opt_state.replace(
                actor=self.optimizer.init(pop_actor_params),
            )

            td3_metrics, agent_state, new_opt_state = self._rl_update(
                agent_state, new_opt_state, replay_buffer_state, learn_key
            )

            # drop the actors' opt_state
            opt_state = opt_state.replace(
                critic=new_opt_state.critic,
            )

            pop_actor_params = agent_state.params.actor_params

        else:
            td3_metrics = None

        # ======== pop rollout ========
        # the trajectory [#pop, T, B, ...]
        # metrics: [#pop, B]
        eval_metrics, trajectory = self._rollout(agent_state, rollout_key)

        fitnesses = eval_metrics.episode_returns.mean(axis=-1)

        replay_buffer_state = self._add_to_replay_buffer(
            replay_buffer_state,
            trajectory,
            eval_metrics.episode_lengths.flatten(),
        )

        train_metrics = POPTrainMetric(
            rb_size=get_buffer_size(replay_buffer_state),
            pop_episode_lengths=eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=eval_metrics.episode_returns.mean(-1),
            rl_metrics=td3_metrics,
            ec_info=PyTreeDict(cov_eps=ec_opt_state.cov_eps),
        )

        # record pop mean, but do not sample new pop:
        ec_metrics, ec_opt_state = self._ec_update(ec_opt_state, fitnesses)

        # calculate the number of timestep
        sampled_timesteps = eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        sampled_episodes = jnp.uint32(self.config.episodes_for_fitness * pop_size)

        # iterations is the number of updates of the agent

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

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        pop_mean_actor_params = state.ec_opt_state.mean

        pop_mean_agent_state = replace_actor_params(
            state.agent_state, pop_mean_actor_params
        )

        key, eval_key, eval_pop_key = jax.random.split(state.key, num=3)

        # [#episodes]
        pop_center_eval_metrics = self.evaluator.evaluate(
            pop_mean_agent_state, eval_key, num_episodes=self.config.eval_episodes
        )

        pop_eval_metrics = self.evaluator.evaluate(
            state.agent_state,
            jax.random.split(eval_pop_key, self.config.num_learning_offspring),
            num_episodes=self.config.eval_episodes,
        )

        eval_metrics = EvaluateMetric(
            pop_center_episode_returns=pop_center_eval_metrics.episode_returns.mean(),
            pop_center_episode_lengths=pop_center_eval_metrics.episode_lengths.mean(),
            pop_episode_returns=pop_eval_metrics.episode_returns.mean(-1),
            pop_episode_lengths=pop_eval_metrics.episode_lengths.mean(-1),
        )

        state = state.replace(key=key)

        return eval_metrics, state

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
                train_metrics_dict["rl_metrics"]["actor_loss"] /= (
                    self.config.num_learning_offspring
                )
                train_metrics_dict["rl_metrics"]["raw_loss_dict"] = jtu.tree_map(
                    get_1d_array_statistics,
                    train_metrics_dict["rl_metrics"]["raw_loss_dict"],
                )

            self.recorder.write(train_metrics_dict, iters)

            std_statistics = get_std_statistics(state.ec_opt_state.variance["params"])
            self.recorder.write({"ec/std": std_statistics}, iters)

            if iters % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)

                eval_metrics_dict = eval_metrics.to_local_dict()

                eval_metrics_dict["pop_episode_returns"] = get_1d_array_statistics(
                    eval_metrics_dict["pop_episode_returns"], histogram=True
                )
                eval_metrics_dict["pop_episode_lengths"] = get_1d_array_statistics(
                    eval_metrics_dict["pop_episode_lengths"], histogram=True
                )

                self.recorder.write(add_prefix(eval_metrics_dict, "eval"), iters)

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)

            self.checkpoint_manager.save(iters, args=ocp.args.StandardSave(state))

        return state
