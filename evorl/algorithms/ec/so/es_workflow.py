import logging
from functools import partial
from omegaconf import DictConfig

import jax
import jax.tree_util as jtu

from evorl.distributed import unpmap
from evorl.agent import Agent, AgentState, AgentStateAxis
from evorl.evaluators import Evaluator
from evorl.metrics import EvaluateMetric, MetricBase
from evorl.types import State
from evorl.envs import Env
from evorl.ec.optimizers import EvoOptimizer
from evorl.recorders import get_1d_array_statistics
from evorl.workflows import ECWorkflowTemplate

logger = logging.getLogger(__name__)


class ESWorkflowTemplate(ECWorkflowTemplate):
    def __init__(
        self,
        *,
        env: Env,
        agent: Agent,
        ec_optimizer: EvoOptimizer,
        ec_evaluator: Evaluator,
        evaluator: Evaluator,
        agent_state_vmap_axes: AgentStateAxis = 0,
        config: DictConfig,
    ):
        super().__init__(
            env=env,
            agent=agent,
            ec_optimizer=ec_optimizer,
            ec_evaluator=ec_evaluator,
            agent_state_vmap_axes=agent_state_vmap_axes,
            config=config,
        )

        self.evaluator = evaluator  # independent evaluator for pop_center

    def _get_pop_center(self, state: State) -> AgentState:
        raise NotImplementedError

    def _record_callback(self, state: State, iters: int) -> None:
        pass

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        """Evaluate the policy with the mean of ES."""
        key, eval_key = jax.random.split(state.key, num=2)

        agent_state = self._get_pop_center(state)

        # [#episodes]
        raw_eval_metrics = self.evaluator.evaluate(
            agent_state, eval_key, num_episodes=self.config.eval_episodes
        )

        eval_metrics = EvaluateMetric(
            episode_returns=raw_eval_metrics.episode_returns.mean(),
            episode_lengths=raw_eval_metrics.episode_lengths.mean(),
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return eval_metrics, state.replace(key=key)

    def learn(self, state: State) -> State:
        start_iteration = unpmap(state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            workflow_metrics = unpmap(workflow_metrics, self.pmap_axis_name)
            self.recorder.write(workflow_metrics.to_local_dict(), iters)

            train_metrics = unpmap(train_metrics, self.pmap_axis_name)
            train_metrics_dict = train_metrics.to_local_dict()
            train_metrics_dict = jtu.tree_map(
                partial(get_1d_array_statistics, histogram=True),
                train_metrics.to_local_dict(),
            )
            self.recorder.write(train_metrics_dict, iters)

            if iters % self.config.eval_interval == 0 or iters == self.config.num_iters:
                eval_metrics, state = self.evaluate(state)
                eval_metrics = unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write(
                    {"eval/pop_center": eval_metrics.to_local_dict()}, iters
                )

            self._record_callback(state, iters)

            self.checkpoint_manager.save(
                iters,
                unpmap(state, self.pmap_axis_name),
                force=i == self.config.num_iters,
            )

    @classmethod
    def enable_jit(cls) -> None:
        super().enable_jit()
        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))

    @classmethod
    def enable_pmap(cls, axis_name) -> None:
        super().enable_pmap(axis_name)
        cls.evaluate = jax.pmap(
            cls.evaluate, axis_name, static_broadcasted_argnums=(0,)
        )
