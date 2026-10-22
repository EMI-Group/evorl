import logging
import math
from functools import partial
from omegaconf import DictConfig

import jax
import jax.tree_util as jtu

from evorl.distributed import POP_AXIS_NAME
from evorl.agent import Agent, AgentState, AgentStateAxis
from evorl.evaluators import Evaluator
from evorl.metrics import EvaluateMetric, MetricBase
from evorl.types import State
from evorl.envs import BaseEnv
from evorl.ec.optimizers import EvoOptimizer
from evorl.recorders import get_1d_array_statistics
from evorl.workflows import ECWorkflowTemplate

logger = logging.getLogger(__name__)


class ESWorkflowTemplate(ECWorkflowTemplate):
    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        """Also split center-policy evaluation across devices."""
        super()._rescale_config(config)
        num_devices = jax.local_device_count()
        if config.num_eval_envs < num_devices or config.num_eval_envs % num_devices:
            raise ValueError(
                f"num_eval_envs({config.num_eval_envs}) must be divisible by num_devices({num_devices})"
            )
        if config.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive")

        eval_iters = math.ceil(config.eval_episodes / config.num_eval_envs)
        if config.eval_episodes % config.num_eval_envs:
            logger.warning(
                f"eval_episodes({config.eval_episodes}) is rounded up to "
                f"{eval_iters * config.num_eval_envs} by the evaluator"
            )
        config.num_eval_envs //= num_devices
        config.eval_episodes = eval_iters * config.num_eval_envs

    def __init__(
        self,
        *,
        env: BaseEnv,
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
        ).all_reduce(axis_name=self.dp_axis_name)

        return eval_metrics, state.replace(key=key)

    def learn(self, state: State) -> State:
        """Run the remaining iterations and return the final workflow state."""
        start_iteration = state.metrics.iterations

        for i in range(start_iteration, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            self.recorder.write(workflow_metrics.to_local_dict(), iters)

            train_metrics_dict = train_metrics.to_local_dict()
            train_metrics_dict = jtu.tree_map(
                partial(get_1d_array_statistics, histogram=True),
                train_metrics.to_local_dict(),
            )
            self.recorder.write(train_metrics_dict, iters)

            if iters % self.config.eval_interval == 0 or iters == self.config.num_iters:
                eval_metrics, state = self.evaluate(state)
                self.recorder.write(
                    {"eval/pop_center": eval_metrics.to_local_dict()}, iters
                )

            self._record_callback(state, iters)

            self.checkpoint_manager.save(
                iters,
                state,
                force=iters == self.config.num_iters,
            )

        return state

    @classmethod
    def enable_jit(cls) -> None:
        super().enable_jit()
        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))

    @classmethod
    def enable_shmap(cls) -> None:
        """Map local computations over workers, keeping self static."""
        super().enable_shmap()
        state_specs = State(
            key=jax.P(POP_AXIS_NAME),
            agent_state=jax.P(),
            ec_opt_state=jax.P(),
            metrics=jax.P(),
            distributed_info=jax.P(POP_AXIS_NAME),
        )

        original_evaluate = cls.evaluate

        def sharded_evaluate(self, state):
            state = state.replace(
                key=state.key[0],
                distributed_info=jax.tree.map(lambda x: x[0], state.distributed_info),
            )
            metrics, state = original_evaluate(self, state)
            return metrics, state.replace(
                key=state.key[None],
                distributed_info=jax.tree.map(
                    lambda x: x[None], state.distributed_info
                ),
            )

        cls.evaluate = jax.shard_map(
            sharded_evaluate,
            in_specs=(None, state_specs),
            out_specs=(jax.P(), state_specs),
        )
