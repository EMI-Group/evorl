from flax import struct

from abc import ABC, abstractmethod
import optax

import chex
from evorl.types import SampleBatch
from evorl.types import (
    Action, Params, PolicyExtraInfo, LossDict
)
from evorl.envs.space import Space
from evorl.utils import running_statistics
from typing import Mapping, Tuple, Union, Any, Optional
import dataclasses

AgentParams = Mapping[str, Params]


@struct.dataclass
class AgentState:
    params: AgentParams
    obs_preprocessor_state: Optional[running_statistics.RunningStatisticsState] = None
    # TODO: define the action_postprocessor_state
    action_postprocessor_state: Any = None
    # opt_state: optax.OptState


# @struct.dataclass
@struct.dataclass
class Agent(ABC):
    """
    Base class for all agents.
    Usage:
    - Store models like actor and critic
    - interactive with environment by compute_actions
    - compute loss by loss
    """
    action_space: Space
    obs_space: Space

    @abstractmethod
    def init(self, key) -> AgentState:
        pass

    @abstractmethod
    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
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
    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        """
            get the best action from the action distribution.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def loss(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
        raise NotImplementedError()

