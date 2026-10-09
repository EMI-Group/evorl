from collections.abc import Iterator
from contextlib import contextmanager
import copy
import logging

import chex
import jax
import optax
from omegaconf import DictConfig, OmegaConf
from typing import Self

from evorl.replay_buffers import (
    AbstractReplayBuffer,
    ReplayBufferState,
    PrioritizedReplayBufferState,
)
from evorl.agent import Agent, AgentState
from evorl.distributed import DP_AXIS_NAME
from evorl.envs import Env
from evorl.evaluators import Evaluator
from evorl.metrics import EvaluateMetric, MetricBase, WorkflowMetric
from evorl.types import State

from .workflow import Workflow

logger = logging.getLogger(__name__)


@contextmanager
def _shard_map_replay_buffer_context(state: State) -> Iterator[State]:
    """Convert per-ring scalar fields at the shard_map boundary.

    Yield a shallow working copy of the workflow state. Update it with the
    computation's returned state before exit; the input state is not mutated.
    """
    buffer_state = state.replay_buffer_state
    buffer_state = buffer_state.replace(
        current_index=buffer_state.current_index.squeeze(0),
        buffer_size=buffer_state.buffer_size.squeeze(0),
    )
    if isinstance(buffer_state, PrioritizedReplayBufferState):
        buffer_state = buffer_state.replace(
            max_priority=buffer_state.max_priority.squeeze(0)
        )
    local_state = state.replace(replay_buffer_state=buffer_state)
    yield local_state
    buffer_state = local_state.replay_buffer_state
    buffer_state = buffer_state.replace(
        current_index=buffer_state.current_index[None],
        buffer_size=buffer_state.buffer_size[None],
    )
    if isinstance(buffer_state, PrioritizedReplayBufferState):
        buffer_state = buffer_state.replace(
            max_priority=buffer_state.max_priority[None]
        )
    local_state.replay_buffer_state = buffer_state


class RLWorkflow(Workflow):
    """Base Workflow for RL algorithms.

    Multi-device states keep a replicated scalar root key. Split off the next
    root, then fold the data-parallel axis index into the consumed subkeys for
    local randomness. The saved next root must stay identical across devices.
    """

    def __init__(self, config: DictConfig):
        """Initialize a RLWorkflow instance.

        Args:
            config: the config object.
        """
        super().__init__(config)

        self.dp_axis_name = None

    @property
    def is_multi_device(self) -> bool:
        """Whether multi-devices training is enabled."""
        return self.dp_axis_name is not None

    @classmethod
    def build_from_config(
        cls,
        config: DictConfig,
        multi_device: bool = False,
        jit: bool = True,
    ) -> Self:
        """Build the rl workflow instance from the config.

        Call on a fresh subclass for each execution configuration, as in
        ``scripts/train.py``. The enable_* class methods transform that subclass.

        Args:
            config: Config of the workflow.
            multi_device: Whether multi-devices training is enabled.
            jit: Whether jit is enabled.
        """
        config = copy.deepcopy(config)  # avoid in-place modification

        devices = jax.local_devices()

        if multi_device:
            OmegaConf.set_readonly(config, False)
            cls._rescale_config(config)

        OmegaConf.set_readonly(config, True)

        workflow = cls._build_from_config(config)
        if multi_device:
            workflow.dp_axis_name = DP_AXIS_NAME
            workflow.mesh = jax.make_mesh(
                (len(devices),),
                (DP_AXIS_NAME,),
                axis_types=(jax.sharding.AxisType.Explicit,),
                devices=devices,
            )
            with jax.set_mesh(workflow.mesh):
                cls.enable_shmap()

        if jit:
            cls.enable_jit()

        return workflow

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        """Customize the process of building the workflow instance from the config.

        Args:
            config: Config of the workflow.

        Returns:
            The created workflow instance.
        """
        raise NotImplementedError

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        """Customize the logic of rescaling the config settings when multi-devices training is enabled.

        When multi_device=True, rescale config settings in-place to match multi-devices.

        Args:
            config: Config of the workflow.
        """
        pass

    def step(self, state: State) -> tuple[MetricBase, State]:
        """Customize the training logic of one iteration.

        Args:
            state: State of the workflow.

        Returns:
            Tuple of (metrics, state).
        """
        raise NotImplementedError

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        """Customize the evaluation logic for the workflow.

        Args:
            state: State of the workflow.
        """
        raise NotImplementedError

    @classmethod
    def enable_jit(cls) -> None:
        """Define which methods should be jitted.

        Compile initialization as well as the training and evaluation methods.
        """
        cls.setup = jax.jit(cls.setup, static_argnums=(0,))
        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls.step = jax.jit(cls.step, static_argnums=(0,))

    @classmethod
    def enable_shmap(cls) -> None:
        """Define distributed method boundaries in the concrete workflow template."""
        raise NotImplementedError


class OnPolicyWorkflow(RLWorkflow):
    """Workflow template for On-Policy RL algorithms.

    This class constructs the template for On-Policy RL algorithms, providing the general `setup()` and `evaluate()` methods.
    """

    def __init__(
        self,
        env: Env,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        evaluator: Evaluator,
        config: DictConfig,
    ):
        """Initialize an OnPolicyWorkflow instance.

        Args:
            env: Environment object.
            agent: Workflow-sepecific agent object.
            optimizer: Optimizer of the agent.
            evaluator: Evaluator object used in self.evaluation().
            config: Config of the workflow.
        """
        super().__init__(config)

        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.evaluator = evaluator

    @classmethod
    def enable_shmap(cls) -> None:
        """Map local computations over workers, keeping self static."""
        state_specs = State(
            key=jax.P(),
            metrics=jax.P(),
            agent_state=jax.P(),
            env_state=jax.P(DP_AXIS_NAME),
            opt_state=jax.P(),
        )

        cls.step = jax.shard_map(
            cls.step,
            in_specs=(None, state_specs),
            out_specs=(jax.P(), state_specs),
            check_vma=False,
        )
        cls.evaluate = jax.shard_map(
            cls.evaluate,
            in_specs=(None, state_specs),
            out_specs=(jax.P(), state_specs),
            check_vma=False,
        )

    def _setup_agent_and_optimizer(
        self, key: jax.Array
    ) -> tuple[AgentState, chex.ArrayTree]:
        """Setup Agent and Optimizer states.

        Args:
            key: JAX PRNGKey.

        Returns:
            Tuple of (agent_state, opt_state)
        """
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        opt_state = self.optimizer.init(agent_state.params)
        return agent_state, opt_state

    def _setup_workflow_metrics(self) -> MetricBase:
        """Define Workflow metrics."""
        return WorkflowMetric()

    def setup(self, key: jax.Array) -> State:
        key, agent_key, env_key = jax.random.split(key, 3)

        agent_state, opt_state = self._setup_agent_and_optimizer(agent_key)
        workflow_metrics = self._setup_workflow_metrics()

        if self.is_multi_device:
            workflow_metrics, agent_state, opt_state = jax.device_put(
                (workflow_metrics, agent_state, opt_state), jax.P()
            )

            key, env_key = jax.device_put((key, env_key), jax.P())

            # reset already batches the local environments. Concatenate those
            # batches along the environment axis, without adding a device axis.
            def reset_env(env_key):
                env_key = jax.random.fold_in(
                    env_key, jax.lax.axis_index(self.dp_axis_name)
                )
                return self.env.reset(env_key)

            env_reset_fn = jax.shard_map(
                reset_env,
                in_specs=jax.P(),
                out_specs=jax.P(self.dp_axis_name),
                check_vma=False,
            )
            env_state = env_reset_fn(env_key)
        else:
            env_state = self.env.reset(env_key)

        return State(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state,
        )

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        key, eval_key = jax.random.split(state.key, num=2)
        if self.is_multi_device:
            eval_key = jax.random.fold_in(
                eval_key, jax.lax.axis_index(self.dp_axis_name)
            )

        # [#episodes]
        raw_eval_metrics = self.evaluator.evaluate(
            state.agent_state, eval_key, num_episodes=self.config.eval_episodes
        )

        eval_metrics = EvaluateMetric(
            episode_returns=raw_eval_metrics.episode_returns.mean(),
            episode_lengths=raw_eval_metrics.episode_lengths.mean(),
        ).all_reduce(dp_axis_name=self.dp_axis_name)

        state = state.replace(key=key)
        return eval_metrics, state


class OffPolicyWorkflow(RLWorkflow):
    """Workflow template for Off-Policy RL algorithms.

    This class constructs the template for Off-Policy RL algorithms, providing the general `setup()` and `evaluate()` methods.
    """

    def __init__(
        self,
        env: Env,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        evaluator: Evaluator,
        replay_buffer: AbstractReplayBuffer,
        config: DictConfig,
    ):
        """Initialize an OffPolicyWorkflow instance.

        Args:
            env: Environment object.
            agent: Workflow-sepecific agent object.
            optimizer: Optimizer of the agent.
            evaluator: Evaluator object used in self.evaluation().
            replay_buffer: ReplayBuffer object.
            config: Config of the workflow.
        """
        super().__init__(config)

        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.replay_buffer = replay_buffer

    @classmethod
    def enable_shmap(cls) -> None:
        """Map local computations over workers, keeping self static."""
        state_specs = State(
            key=jax.P(),
            metrics=jax.P(),
            agent_state=jax.P(),
            env_state=jax.P(DP_AXIS_NAME),
            opt_state=jax.P(),
            replay_buffer_state=jax.P(DP_AXIS_NAME),
        )

        step = cls.step

        def sharded_step(self, state):
            with _shard_map_replay_buffer_context(state) as local_state:
                metrics, next_state = step(self, local_state)
                local_state.update(next_state)
            return metrics, local_state

        cls.step = jax.shard_map(
            sharded_step,
            in_specs=(None, state_specs),
            out_specs=(jax.P(), state_specs),
            check_vma=False,
        )
        # Evaluation passes the buffer through without reading or updating it.
        cls.evaluate = jax.shard_map(
            cls.evaluate,
            in_specs=(None, state_specs),
            out_specs=(jax.P(), state_specs),
            check_vma=False,
        )

        _postsetup_replaybuffer = cls._postsetup_replaybuffer

        def sharded_postsetup_replaybuffer(self, state):
            with _shard_map_replay_buffer_context(state) as local_state:
                next_state = _postsetup_replaybuffer(self, local_state)
                local_state.update(next_state)
            return local_state

        cls._postsetup_replaybuffer = jax.shard_map(
            sharded_postsetup_replaybuffer,
            in_specs=(None, state_specs),
            out_specs=state_specs,
            check_vma=False,
        )

    def _setup_agent_and_optimizer(
        self, key: jax.Array
    ) -> tuple[AgentState, chex.ArrayTree]:
        """Setup Agent and Optimizer states.

        Args:
            key: JAX PRNGKey.

        Returns:
            Tuple of (agent_state, opt_state).
        """
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        opt_state = self.optimizer.init(agent_state.params)
        return agent_state, opt_state

    def _setup_workflow_metrics(self) -> MetricBase:
        """Define Workflow metrics."""
        return WorkflowMetric()

    def _setup_replaybuffer(self, key: jax.Array) -> ReplayBufferState:
        """Setup ReplayBuffer state."""
        raise NotImplementedError

    def _postsetup_replaybuffer(self, state: State) -> State:
        """Post-setup ReplayBuffer state before training."""
        return state

    def setup(self, key: jax.Array) -> State:
        key, agent_key, env_key, rb_key = jax.random.split(key, 4)

        agent_state, opt_state = self._setup_agent_and_optimizer(agent_key)
        workflow_metrics = self._setup_workflow_metrics()

        if self.is_multi_device:
            workflow_metrics, agent_state, opt_state = jax.device_put(
                (workflow_metrics, agent_state, opt_state), jax.P()
            )

            key, env_key, rb_key = jax.device_put((key, env_key, rb_key), jax.P())

            # reset already batches the local environments. Concatenate those
            # batches along the environment axis, without adding a device axis.
            def reset_env(env_key):
                env_key = jax.random.fold_in(
                    env_key, jax.lax.axis_index(self.dp_axis_name)
                )
                return self.env.reset(env_key)

            env_reset_fn = jax.shard_map(
                reset_env,
                in_specs=jax.P(),
                out_specs=jax.P(self.dp_axis_name),
                check_vma=False,
            )
            env_state = env_reset_fn(env_key)

            def setup_replaybuffer(rb_key):
                rb_key = jax.random.fold_in(
                    rb_key, jax.lax.axis_index(self.dp_axis_name)
                )
                buffer_state = self._setup_replaybuffer(rb_key)
                buffer_state = buffer_state.replace(
                    current_index=buffer_state.current_index[None],
                    buffer_size=buffer_state.buffer_size[None],
                )
                if isinstance(buffer_state, PrioritizedReplayBufferState):
                    buffer_state = buffer_state.replace(
                        max_priority=buffer_state.max_priority[None]
                    )
                return buffer_state

            setup_rb_fn = jax.shard_map(
                setup_replaybuffer,
                in_specs=jax.P(),
                out_specs=jax.P(self.dp_axis_name),
                check_vma=False,
            )
            replay_buffer_state = setup_rb_fn(rb_key)
        else:
            env_state = self.env.reset(env_key)
            replay_buffer_state = self._setup_replaybuffer(rb_key)

        state = State(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state,
            replay_buffer_state=replay_buffer_state,
        )

        logger.info("Start replay buffer post-setup")
        state = self._postsetup_replaybuffer(state)

        logger.info("Complete replay buffer post-setup")

        return state

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        key, eval_key = jax.random.split(state.key, num=2)
        if self.is_multi_device:
            eval_key = jax.random.fold_in(
                eval_key, jax.lax.axis_index(self.dp_axis_name)
            )

        # [#episodes]
        raw_eval_metrics = self.evaluator.evaluate(
            state.agent_state, eval_key, num_episodes=self.config.eval_episodes
        )

        eval_metrics = EvaluateMetric(
            episode_returns=raw_eval_metrics.episode_returns.mean(),
            episode_lengths=raw_eval_metrics.episode_lengths.mean(),
        ).all_reduce(dp_axis_name=self.dp_axis_name)

        state = state.replace(key=key)
        return eval_metrics, state
