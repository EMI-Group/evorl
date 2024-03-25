import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig
import chex

from .workflow import Workflow
from evorl.agents import Agent
from evorl.envs import Env
from evorl.evaluator import Evaluator
from evorl.distributed import PMAP_AXIS_NAME
from evorl.types import TrainMetric, SampleBatch
from typing import Any, Callable, Sequence, Optional

from evox import State


class RLWorkflow(Workflow):
    def __init__(
        self,
        config: DictConfig,
        env: Env,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        evaluator: Evaluator
    ):
        super(RLWorkflow, self).__init__()
        self.config = config
        self.agent = agent
        self.env = env  # batched env
        self.optimizer = optimizer
        self.evaluator = evaluator

        self.pmap_axis_name = None

    def evaluate(self, state: State) -> State:
        """
            run the complete evaluation process.
            Note: this is designed for the non pure function. Don't wrap it with jit.
        """
        raise NotImplementedError

    def enable_multi_devices(self, state: State, devices: Optional[Sequence[jax.Device]]) -> State:
        """
            Enable multi devices setup for the workflow.
            Note: this implicitly enables jit for step() and evaluate()
        """
        if devices is None:
            devices = jax.local_devices()

        self.pmap_axis_name = PMAP_AXIS_NAME
        self.step = jax.pmap(self.step, axis_name=PMAP_AXIS_NAME)
        self.evaluate = jax.pmap(self.evaluate, axis_name=PMAP_AXIS_NAME)

        key = state.key
        state = jax.device_put_replicated(state, devices)
        # this ensures randomness in different devices
        key_devices = jax.device_put_sharded(
            tuple(_key for _key in jax.random.split(key, len(devices))), 
            devices)
        state = state.update(key=key_devices)

        return state

    def enable_jit(self) -> None:
        self.evaluate = jax.jit(self.evaluate)
        super().enable_jit()


class OnPolicyRLWorkflow(RLWorkflow):
    def __init__(
        self,
        config: DictConfig,
        env: Env,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        evaluator: Evaluator
    ):
        super(OnPolicyRLWorkflow, self).__init__(
            config=config,
            env=env,
            agent=agent,
            optimizer=optimizer,
            evaluator=evaluator
        )

    def setup(self, key):
        key, agent_key, env_key = jax.random.split(key, 3)
        agent_state = self.agent.init(agent_key)

        # Note: not need for evaluator state

        return State(
            key=key,
            metric=TrainMetric(),
            agent_state=agent_state,
            env_state=self.env.reset(env_key),
            opt_state=self.optimizer.init(agent_state.params)
        )


class OffPolicyRLWorkflow(RLWorkflow):
    def __init__(
        self,
        config: DictConfig,
        env: Env,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        evaluator: Evaluator,
        replay_buffer: Any,
        replay_buffer_init_fn: Callable[[Any, chex.PRNGKey], chex.ArrayTree]
    ):
        super(OffPolicyRLWorkflow, self).__init__(
            config=config,
            agent=agent,
            env=env,
            optimizer=optimizer,
            evaluator=evaluator
        )
        self.replay_buffer = replay_buffer
        self._init_replay_buffer = replay_buffer_init_fn

    def setup(self, key):
        key, agent_key, env_key, buffer_key = jax.random.split(key, 4)
        agent_state = self.agent.init(agent_key)

        env_state = self.env.reset(env_key)

        replay_buffer_state = self._init_replay_buffer(
            self.replay_buffer, buffer_key)

        return State(
            key=key,
            metric=TrainMetric(),
            replay_buffer_state=replay_buffer_state,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=self.optimizer.init(agent_state.params)
        )
