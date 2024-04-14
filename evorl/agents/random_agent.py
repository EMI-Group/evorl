
import jax
import jax.numpy as jnp
import chex

from evorl.types import PolicyExtraInfo
from evorl.sample_batch import SampleBatch
from typing import Tuple
from evorl.types import (
    LossDict, Action, PolicyExtraInfo, PyTreeDict
)
from .agent import Agent, AgentState

import dataclasses


class DebugRandomAgent(Agent):
    """
        An agent that takes random actions.
        Used for testing and debugging.
    """
    def init(self, key:chex.PRNGKey) -> AgentState:
        return AgentState(
            params={}
        )

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        batch_shapes = sample_batch.obs.shape[:1]
        actions = self.action_space.sample(key)
        actions =  jnp.broadcast_to(actions, batch_shapes+actions.shape)
        return actions, PyTreeDict()
    
    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        return self.compute_actions(agent_state, sample_batch, key)
    
    def loss(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
        return PyTreeDict(
            loss=jnp.zeros(())
        )
    
class RandomAgent(Agent):
    """
        An agent that takes random actions.
    """
    def init(self, key:chex.PRNGKey) -> AgentState:
        return AgentState(
            params={}
        )

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        batch_size = sample_batch.obs.shape[0]
        actions = jax.vmap(self.action_space.sample)(jax.random.split(key, batch_size))
        return actions, PyTreeDict()
    
    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        return self.compute_actions(agent_state, sample_batch, key)
    
    def loss(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
        raise NotImplementedError("RandomAgent does not have a loss function")
        
        