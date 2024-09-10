from collections.abc import Callable, Sequence
import logging
from functools import partial

from evox import Algorithm, Problem
import jax
import jax.tree_util as jtu
import orbax.checkpoint as ocp

from evorl.distributed import tree_unpmap
from evorl.agent import Agent, AgentState
from evorl.evaluator import Evaluator
from evorl.metrics import EvaluateMetric, MetricBase
from evorl.types import State
from evorl.workflows import ECWorkflow
from evorl.recorders import get_1d_array_statistics

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ESBaseWorkflow(ECWorkflow):
    def __init__(
        self,
        config: DictConfig,
        agent: Agent,
        evaluator: Evaluator,
        algorithm: Algorithm,
        problem: Problem,
        opt_direction: str | Sequence[str] = "max",
        candidate_transforms: Sequence[Callable] = (),
        fitness_transforms: Sequence[Callable] = (),
    ):
        super().__init__(
            config=config,
            agent=agent,
            algorithm=algorithm,
            problem=problem,
            opt_direction=opt_direction,
            candidate_transforms=candidate_transforms,
            fitness_transforms=fitness_transforms,
        )

        # An extra evalutor for pop_center
        self.evaluator = evaluator

    def evaluate(self, state: State) -> tuple[EvaluateMetric, State]:
        raise NotImplementedError

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


class ESWorkflowTemplate(ESBaseWorkflow):
    @staticmethod
    def _rescale_config(config: DictConfig) -> None:
        num_devices = jax.device_count()

        if config.num_envs % num_devices != 0:
            logger.warning(
                f"num_envs ({config.num_envs}) must be divisible by the number of devices ({num_devices}), "
                f"rescale eval_episodes to {config.eval_episodes // num_devices}"
            )

        config.eval_episodes = config.eval_episodes // num_devices

    def _get_pop_center(self, state: State) -> AgentState:
        raise NotImplementedError

    def evaluate(self, state: State) -> tuple[EvaluateMetric, State]:
        """Evaluate the policy with the mean of CMAES"""
        key, eval_key = jax.random.split(state.key, num=2)

        agent_state = self._get_pop_center(state)

        # [#episodes]
        raw_eval_metrics = self.evaluator.evaluate(
            agent_state, num_episodes=self.config.eval_episodes, key=eval_key
        )

        eval_metrics = EvaluateMetric(
            episode_returns=raw_eval_metrics.episode_returns.mean(),
            episode_lengths=raw_eval_metrics.episode_lengths.mean(),
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return eval_metrics, state.replace(key=key)

    def _record_callback(
        self,
        state: State,
        train_metrics: MetricBase,
        eval_metrics: MetricBase = None,
        iters: int = 0,
    ) -> None:
        """
        Add some customized metrics
        """
        pass

    def learn(self, state: State) -> State:
        start_iteration = tree_unpmap(state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            workflow_metrics = tree_unpmap(workflow_metrics, self.pmap_axis_name)
            self.recorder.write(workflow_metrics.to_local_dict(), iters)

            train_metrics = tree_unpmap(train_metrics, self.pmap_axis_name)
            train_metrics_dict = train_metrics.to_local_dict()
            train_metrics_dict = jtu.tree_map(
                partial(get_1d_array_statistics, histogram=True),
                train_metrics.to_local_dict(),
            )
            self.recorder.write(train_metrics_dict, iters)

            eval_metrics, state = self.evaluate(state)
            eval_metrics = tree_unpmap(eval_metrics, self.pmap_axis_name)
            self.recorder.write(
                {"eval/pop_center": eval_metrics.to_local_dict()}, iters
            )

            self._record_callback(state, train_metrics, eval_metrics, iters)

            self.checkpoint_manager.save(
                iters,
                args=ocp.args.StandardSave(
                    tree_unpmap(state, self.pmap_axis_name),
                ),
            )
