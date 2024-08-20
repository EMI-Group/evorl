from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import chex

from evorl.envs import Space
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action,
    LossDict,
    Params,
    PolicyExtraInfo,
    PyTreeData,
    PyTreeNode,
)
from evorl.utils import running_statistics

AgentParams = Mapping[str, Params]


class AgentState(PyTreeData):
    params: AgentParams
    obs_preprocessor_state: running_statistics.RunningStatisticsState | None = None
    # TODO: define the action_postprocessor_state
    action_postprocessor_state: Any = None


AgentActionFn = Callable[
    [AgentState, SampleBatch, chex.PRNGKey], tuple[Action, PolicyExtraInfo]
]


class Agent(PyTreeNode, metaclass=ABCMeta):
    """
    Base class for all agents.
    Usage:
    - Store models like actor and critic
    - interactive with environment by compute_actions
    - compute loss by loss
    """

    @abstractmethod
    def init(self, obs_space: Space, action_space: Space, key) -> AgentState:
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


class LossFn(Protocol):
    def __call__(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        """
        The type for the loss function for the model.
        In some case, a single loss function is not enough. For example, DDPG has two loss functions: actor_loss and critic_loss.
        """
        pass
