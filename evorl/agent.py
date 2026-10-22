from abc import ABCMeta, abstractmethod
from typing import Any, Protocol

import jax
import jax.tree_util as jtu
import chex
from flax import nnx
from evorl.utils.jax_utils import rng_split_by_shape

from evorl.envs import Space, is_leaf_space
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action,
    Axis,
    LossDict,
    PolicyExtraInfo,
    PyTreeData,
    PyTreeNode,
    PyTreeDict,
)


class AgentState(PyTreeData):
    """State of the agent.

    Attributes:
        model_state: Complete NNX network graph and variables. The graph is
            static PyTree metadata; parameters and other variables are leaves.
        obs_preprocessor_state: The state of the observation preprocessor.
        action_postprocessor_state: The state of the action postprocessor.
        extra_state: Other dynamic algorithm data, including tunable
            hyperparameters. Network auxiliary variables belong in `model_state`.
    """

    model_state: nnx.GraphState | None = None
    obs_preprocessor_state: Any = None
    # TODO: define the action_postprocessor_state
    action_postprocessor_state: Any = None
    extra_state: Any = None


AgentStateAxis = AgentState | Axis


class ObsPreprocessorFn(Protocol):
    """The type of the observation preprocessor function."""

    def __call__(self, obs: jax.Array, *args: Any, **kwds: Any) -> jax.Array:
        return obs


class LossFn(Protocol):
    """Compute loss metrics from an agent's working training state.

    Model auxiliary variables may change during the forward pass. The working
    state shares those variables with the models managed by the NNX gradient
    transform; the workflow saves their updated snapshot after training.
    """

    def __call__(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> LossDict:
        pass


class AgentActionFn(Protocol):
    """The type of the agent's action function."""

    def __call__(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        pass


class Agent(PyTreeNode, metaclass=ABCMeta):
    """Agent Interface.

    The responsibilities of an Agent:

    - Keep fixed network and algorithm configuration, and create NNX networks
      in `init()`. Complete network snapshots live in `AgentState`.
    - Interact with environment by `compute_actions()` or `evaluate_actions()`.
    - Compute algorithm-specific losses (optional).
    """

    def get_models(self, agent_state: AgentState, *, copy: bool = False) -> nnx.Module:
        """Restore the NNX models represented by an agent state.

        Losses share variables with the working models passed through
        `nnx.value_and_grad`. Their temporary `AgentState` is created with
        `nnx.unpack(models)`, so forward-pass state updates reach those models
        without an additional return value. Workflows first restore an isolated
        copy of the input models and save it after the optimizer update.

        Sharing applies to existing variable values, updated with `variable[...]`.
        Module attribute replacement and structural changes are not shared.

        Args:
            agent_state: State containing a complete NNX network snapshot.
            copy: Whether to copy variable objects. Use `True` for an isolated
                model, including rollout and evaluation calls that must preserve
                their input state. The default shares variables for training.

        Returns:
            Restored NNX models, with shared or copied variable objects.
        """
        return nnx.merge(agent_state.model_state, copy=copy)

    @abstractmethod
    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        """Create a new state from fixed configuration without modifying self.

        Capture the spaces when jitting initialization so dimensions such as
        the number of discrete actions remain concrete during construction.
        """

    @abstractmethod
    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
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
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
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

    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        extra_state = PyTreeDict(
            action_space=action_space,
            obs_space=obs_space,
        )
        return AgentState(extra_state=extra_state)

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        obs_space = agent_state.extra_state.obs_space
        action_space = agent_state.extra_state.action_space

        _obs = jtu.tree_leaves(sample_batch.obs)[0]
        _obs_space = jtu.tree_leaves(obs_space, is_leaf=is_leaf_space)[0]
        batch_shapes = _obs.shape[: _obs.ndim - len(_obs_space.shape)]

        chex.assert_tree_shape_prefix(sample_batch.obs, batch_shapes)

        action_sample_fn = action_space.sample
        for _ in range(len(batch_shapes)):
            action_sample_fn = jax.vmap(action_sample_fn)

        action_keys = rng_split_by_shape(key, batch_shapes)

        actions = action_sample_fn(action_keys)
        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        return self.compute_actions(agent_state, sample_batch, key)
