
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


class RandomAgent:
    """
        A duck-type agent that takes random actions.
        Used for testing and debugging.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.obs_space = self.env.obs_space

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        obs_ndims = len(self.env.obs_space.shape)
        batch_shapes = sample_batch.obs.shape[:-obs_ndims]
        actions = self.action_space.sample(batch_shapes)
        actions =  jnp.broadcast_to(actions, batch_shapes+actions.shape)
        return actions, {}
    

        
        