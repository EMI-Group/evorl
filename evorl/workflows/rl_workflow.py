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
from evorl.types import TrainMetric, SampleBatch
from evorl.utils.jax_utils import jit_method, pmap_method
from typing import Any, Callable, Sequence, Optional, TypeVar
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
    def build_from_config(cls, config, enable_multi_devices: bool = False, devices: Optional[Sequence[jax.Device]] = None):
        config = copy.deepcopy(config) # avoid in-place modification
        devices = jax.local_devices() if devices is None else devices

        if enable_multi_devices:
            cls.step = jax.pmap(
                cls.step, axis_name=PMAP_AXIS_NAME, static_broadcasted_argnums=(0,))
            cls.evaluate = jax.pmap(
                cls.evaluate, axis_name=PMAP_AXIS_NAME, static_broadcasted_argnums=(0,))
            OmegaConf.set_readonly(config, False)
            config = cls._rescale_config(config, devices)

        OmegaConf.set_readonly(config, True)

        workflow = cls._build_from_config(config)
        if enable_multi_devices:
            workflow.pmap_axis_name = PMAP_AXIS_NAME
            workflow.devices = devices

        return workflow

    @classmethod
    def _build_from_config(cls, config) -> Self:
        raise NotImplementedError

    @staticmethod
    def _rescale_config(config, devices) -> None:
        """
            When enable_multi_devices=True, rescale config settings to match multi-devices
        """
        raise NotImplementedError

    def evaluate(self, state: State) -> State:
        """
            run the complete evaluation process.
            Note: this is designed for the non pure function. Don't wrap it with jit.
        """
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

    def _setup_train_metrics(self) -> TrainMetric:
        return TrainMetric()

    def setup(self, key):
        key, agent_key, env_key = jax.random.split(key, 3)

        agent_state = self.agent.init(agent_key)

        train_metrics = self._setup_train_metrics()
        opt_state = self.optimizer.init(agent_state.params)

        if self.enable_multi_devices:
            devices = self.devices

            train_metrics, agent_state, opt_state = jax.device_put_replicated(
                (train_metrics, agent_state, opt_state), devices)

            # key and env_state should be different over devices
            key = split_key_to_devices(key, devices)

            env_key = split_key_to_devices(env_key, devices)
            env_state = jax.pmap(
                self.env.reset, axis_name=self.pmap_axis_name)(env_key)
        else:
            env_state = self.env.reset(env_key)

        return State(
            key=key,
            train_metrics=train_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state
        )


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

    def _setup_train_metrics(self) -> TrainMetric:
        return TrainMetric()

    def setup(self, key):
        key, agent_key, env_key, buffer_key = jax.random.split(key, 4)

        agent_state = self.agent.init(agent_key)

        train_metrics = self._setup_train_metrics()
        opt_state = self.optimizer.init(agent_state.params)

        replay_buffer_state = self._init_replay_buffer(
            self.replay_buffer, buffer_key)

        if self.enable_multi_devices:
            devices = self.devices

            train_metrics, agent_state, opt_state, replay_buffer_state = jax.device_put_replicated(
                (train_metrics, agent_state, opt_state, replay_buffer_state), devices)

            # key and env_state should be different over devices
            key = split_key_to_devices(key, devices)

            env_key = split_key_to_devices(env_key, devices)
            env_state = jax.pmap(
                self.env.reset, axis_name=self.pmap_axis_name
            )(env_key)
        else:
            env_state = self.env.reset(env_key)

        return State(
            key=key,
            train_metrics=train_metrics,
            replay_buffer_state=replay_buffer_state,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state
        )
