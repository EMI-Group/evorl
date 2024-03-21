
import jax
import jax.numpy as jnp
import chex
from flax import struct

from evorl.types import PolicyExtraInfo, SampleBatch
from typing import Tuple, Sequence
from evorl.types import (
    EnvLike, LossDict, Action, Params, PolicyExtraInfo, EnvState,
    Observation
)
from .agent import Agent, AgentState

@struct.dataclass
class RandomAgent(Agent):
    """
        An agent that takes random actions.
        Used for testing and debugging.
    """
    def init(self, key:chex.PRNGKey) -> AgentState:
        return AgentState(
            params={}
        )


    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        batch_shapes = (sample_batch.obs.shape[0],)
        actions = self.action_space.sample(key)
        actions =  jnp.broadcast_to(actions, batch_shapes+actions.shape)
        return actions, {}
    
    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Action:
        return self.compute_actions(agent_state, sample_batch, key)
    
    def loss(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
        return dict(
            loss=jnp.zeros(())
        )
    

        
        