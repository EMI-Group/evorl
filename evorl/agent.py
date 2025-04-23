from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol

import jax
import jax.tree_util as jtu
import chex
import numpy as np

from evorl.envs import Space, is_leaf_space
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action,
    Axis,
    LossDict,
    Params,
    PolicyExtraInfo,
    PyTreeData,
    PyTreeNode,
    PyTreeDict,
)


class AgentState(PyTreeData):
    """State of the agent.

    Attributes:
        params: The network parameters of the agent.
        obs_preprocessor_state: The state of the observation preprocessor.
        action_postprocessor_state: The state of the action postprocessor.
        extra_state: Extra state of the agent.
    """

    params: Mapping[str, Params]
    obs_preprocessor_state: Any = None
    # TODO: define the action_postprocessor_state
    action_postprocessor_state: Any = None
    extra_state: Any = None


AgentStateAxis = AgentState | Axis


class ObsPreprocessorFn(Protocol):
    """The type of the observation preprocessor function."""

    def __call__(self, obs: chex.Array, *args: Any, **kwds: Any) -> chex.Array:
        return obs


class LossFn(Protocol):
    """The type of the agent's loss function.

    In some case, a single loss function is not enough. For example, DDPG has two loss functions: actor_loss and critic_loss.
    """

    def __call__(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        pass


class AgentActionFn(Protocol):
    """The type of the agent's action function."""

    def __call__(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        pass


class Agent(PyTreeNode, metaclass=ABCMeta):
    """Agent Interface.

    The responsibilities of an Agent:

    - Store models like actor and critic.
    - Interact with environment by `compute_actions()` or `evaluate_actions()`.
    - Compute algorithm-specific losses (optional).
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
        """Get actions from the policy model + add exploraton noise.

        This method is exclusively used for rollout.

        Args:
            agent_state: the state of the agent.
            sample_batch: Previous Transition data. Usually only contrains `obs`.
            key: JAX PRNGKey.

        Return:
            A tuple (action, policy_extra_info), policy_extra_info is a dict containing extra information about the policy, such as the current hidden state of RNN.
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        """Get the best action from the action distribution.

        This method is exclusively used for evaluation.

        Args:
            agent_state: the state of the agent.
            sample_batch: Previous Transition data. Usually only contrains `obs`.
            key: JAX PRNGKey.

        Return:
            A tuple (action, policy_extra_info), policy_extra_info is a dict containing extra information about the policy, such as the current hidden state of RNN.

        """
        raise NotImplementedError()


class RandomAgent(Agent):
    """An agent that takes uniform random actions."""

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

        _obs = jtu.tree_leaves(sample_batch.obs)[0]
        _obs_space = jtu.tree_leaves(obs_space, is_leaf=is_leaf_space)[0]
        batch_shapes = _obs.shape[: -len(_obs_space.shape)]

        chex.assert_tree_shape_prefix(sample_batch.obs, batch_shapes)

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
