import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig, OmegaConf
import chex
import copy

from .workflow import Workflow
from evorl.agents import Agent
from evorl.envs import Env
from evorl.evaluator import Evaluator
from evorl.distributed import PMAP_AXIS_NAME, split_key_to_devices
from evorl.types import TrainMetric, EvaluateMetric, WorkflowMetric
from typing import Any, Callable, Sequence, Optional, Tuple
from typing_extensions import (
    Self  # pytype: disable=not-supported-yet
)
from evox import State


class RLWorkflow(Workflow):
    def __init__(
        self,
        config: DictConfig,
    ):
        """
            config:
            devices: a single device or a list of devices.
        """
        super(RLWorkflow, self).__init__()
        self.config = config
        self.pmap_axis_name = None
        self.devices = jax.local_devices()[:1]

    @property
    def enable_multi_devices(self) -> bool:
        return self.pmap_axis_name is not None

    @classmethod
    def build_from_config(cls, config: DictConfig, enable_multi_devices: bool = False, devices: Optional[Sequence[jax.Device]] = None):
        config = copy.deepcopy(config)  # avoid in-place modification
        if devices is None:
            devices = jax.local_devices()

        if enable_multi_devices:
            cls.step = jax.pmap(
                cls.step, axis_name=PMAP_AXIS_NAME, static_broadcasted_argnums=(0,))
            cls.evaluate = jax.pmap(
                cls.evaluate, axis_name=PMAP_AXIS_NAME, static_broadcasted_argnums=(0,))
            OmegaConf.set_readonly(config, False)
            cls._rescale_config(config, devices)

        OmegaConf.set_readonly(config, True)

        workflow = cls._build_from_config(config)
        if enable_multi_devices:
            workflow.pmap_axis_name = PMAP_AXIS_NAME
            workflow.devices = devices

        return workflow

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        raise NotImplementedError

    @staticmethod
    def _rescale_config(config: DictConfig, devices) -> None:
        """
            When enable_multi_devices=True, rescale config settings to match multi-devices
        """
        raise NotImplementedError

    def _setup_workflow_metrics(self) -> TrainMetric:
        """
            Customize the workflow metrics.
        """
        return WorkflowMetric()

    def step(self, key: chex.PRNGKey) -> Tuple[TrainMetric, State]:
        raise NotImplementedError

    def evaluate(self, state: State) -> Tuple[EvaluateMetric, State]:
        raise NotImplementedError

    @classmethod
    def enable_jit(cls) -> None:
        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        super().enable_jit()


class OnPolicyRLWorkflow(RLWorkflow):
    def __init__(
        self,
        env: Env,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        evaluator: Evaluator,
        config: DictConfig
    ):
        super(OnPolicyRLWorkflow, self).__init__(config)

        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.evaluator = evaluator

    def setup(self, key: chex.PRNGKey) -> State:
        key, agent_key, env_key = jax.random.split(key, 3)

        agent_state = self.agent.init(agent_key)

        workflow_metrics = self._setup_workflow_metrics()
        opt_state = self.optimizer.init(agent_state.params)

        if self.enable_multi_devices:
            workflow_metrics, agent_state, opt_state = \
                jax.device_put_replicated(
                    (workflow_metrics, agent_state, opt_state),
                    self.devices
                )

            # key and env_state should be different over devices
            key = split_key_to_devices(key, self.devices)

            env_key = split_key_to_devices(env_key, self.devices)
            env_state = jax.pmap(
                self.env.reset, axis_name=self.pmap_axis_name)(env_key)
        else:
            env_state = self.env.reset(env_key)

        return State(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state
        )

    def evaluate(self, state: State) -> Tuple[EvaluateMetric, State]:
        key, eval_key = jax.random.split(state.key, num=2)

        # [#episodes]
        raw_eval_metrics = self.evaluator.evaluate(
            state.agent_state,
            num_episodes=self.config.eval_episodes,
            key=eval_key
        )

        eval_metrics = EvaluateMetric(
            discount_returns=raw_eval_metrics.discount_returns.mean(),
            episode_lengths=raw_eval_metrics.episode_lengths.mean()
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        state = state.update(key=key)
        return eval_metrics, state


class OffPolicyRLWorkflow(RLWorkflow):
    def __init__(
        self,
        env: Env,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        evaluator: Evaluator,
        replay_buffer: Any,
        replay_buffer_init_fn: Callable[[Any, chex.PRNGKey], chex.ArrayTree],
        config: DictConfig,
    ):
        super(OffPolicyRLWorkflow, self).__init__(config)

        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.evaluator = evaluator

        self.replay_buffer = replay_buffer
        self._init_replay_buffer = replay_buffer_init_fn

    def setup(self, key: chex.PRNGKey) -> State:
        key, agent_key, env_key, buffer_key = jax.random.split(key, 4)

        agent_state = self.agent.init(agent_key)

        workflow_metrics = self._setup_workflow_metrics()
        opt_state = self.optimizer.init(agent_state.params)

        replay_buffer_state = self._init_replay_buffer(
            self.replay_buffer, buffer_key)

        if self.enable_multi_devices:
            workflow_metrics, agent_state, opt_state, replay_buffer_state = \
                jax.device_put_replicated(
                    (workflow_metrics, agent_state, opt_state, replay_buffer_state),
                    self.devices
                )

            # key and env_state should be different over devices
            key = split_key_to_devices(key, self.devices)

            env_key = split_key_to_devices(env_key, self.devices)
            env_state = jax.pmap(
                self.env.reset, axis_name=self.pmap_axis_name
            )(env_key)
        else:
            env_state = self.env.reset(env_key)

        return State(
            key=key,
            metrics=workflow_metrics,
            replay_buffer_state=replay_buffer_state,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state
        )

    def evaluate(self, state: State) -> Tuple[EvaluateMetric, State]:
        key, eval_key = jax.random.split(state.key, num=2)

        # [#episodes]
        raw_eval_metrics = self.evaluator.evaluate(
            state.agent_state,
            num_episodes=self.config.eval_episodes,
            key=eval_key
        )

        eval_metrics = EvaluateMetric(
            discount_returns=raw_eval_metrics.discount_returns.mean(),
            episode_lengths=raw_eval_metrics.episode_lengths.mean()
        )

        eval_metrics = eval_metrics.all_reduce(
            pmap_axis_name=self.pmap_axis_name)

        state = state.update(key=key)
        return eval_metrics, state
