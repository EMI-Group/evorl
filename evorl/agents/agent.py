from flax import struct

from abc import ABC, abstractmethod
import optax

import chex
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action, Params, PolicyExtraInfo, LossDict
)
from evorl.envs.space import Space
from evorl.utils import running_statistics
from typing import Mapping, Tuple, Union, Any, Optional
import dataclasses

AgentParams = Mapping[str, Params]


class AgentState(struct.PyTreeNode):
    params: AgentParams
    obs_preprocessor_state: Optional[running_statistics.RunningStatisticsState] = None
    # TODO: define the action_postprocessor_state
    action_postprocessor_state: Any = None



@dataclasses.dataclass(unsafe_hash=True)
class Agent(ABC):
    """
    Base class for all agents.
    Usage:
    - Store models like actor and critic
    - interactive with environment by compute_actions
    - compute loss by loss
    """
    action_space: Space = dataclasses.field(hash=False)
    obs_space: Space = dataclasses.field(hash=False)

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
