from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol

import jax
import chex
import numpy as np

from evorl.envs import Space
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action,
    LossDict,
    Params,
    PolicyExtraInfo,
    PyTreeData,
    PyTreeNode,
    PyTreeDict,
)

AgentParams = Mapping[str, Params]


class AgentState(PyTreeData):
    params: AgentParams
    obs_preprocessor_state: Any = None
    # TODO: define the action_postprocessor_state
    action_postprocessor_state: Any = None
    # action_space: Space | None = None
    # obs_space: Space | None = None
    extra_state: Any = None


class Agent(PyTreeNode, metaclass=ABCMeta):
    """
    Base class for all agents.
    Usage:
    - Store models like actor and critic
    - interactive with environment by compute_actions
    - compute loss by loss
    """

    @abstractmethod
    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        pass

    @abstractmethod
    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        """
        get actions from the policy model + add exploraton noise.
        This function is exclusively used for rollout.

        sample_batch: only `obs` field are available.
        key: a single PRNGKey.

        Return:
        - action
        - policy extra info (eg: hidden state of RNN)
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        """
        get the best action from the action distribution.
        """
        raise NotImplementedError()


class ObsPreprocessorFn(Protocol):
    def __call__(self, obs: chex.Array, *args: Any, **kwds: Any) -> chex.Array:
        return obs


class LossFn(Protocol):
    def __call__(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        """
        The type for the loss function for the model.
        In some case, a single loss function is not enough. For example, DDPG has two loss functions: actor_loss and critic_loss.
        """
        pass


class AgentActionFn(Protocol):
    def __call__(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        pass


class RandomAgent(Agent):
    """
    An agent that takes random actions.
    """

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        extra_state = PyTreeDict(
            action_space=action_space,
            obs_space=obs_space,
        )
        return AgentState(params={}, extra_state=extra_state)

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs_space = agent_state.extra_state.obs_space
        action_space = agent_state.extra_state.action_space

        batch_shapes = sample_batch.obs.shape[: -len(obs_space.shape)]

        action_sample_fn = action_space.sample
        for _ in range(len(batch_shapes)):
            action_sample_fn = jax.vmap(action_sample_fn)

        action_keys = jax.random.split(key, np.prod(batch_shapes)).reshape(
            *batch_shapes, 2
        )

        actions = action_sample_fn(action_keys)
        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        return self.compute_actions(agent_state, sample_batch, key)
