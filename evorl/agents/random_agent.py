
import jax
import jax.numpy as jnp
import chex
import numpy as np

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

    def init(self, key: chex.PRNGKey) -> AgentState:
        return AgentState(
            params={}
        )

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        batch_shapes = sample_batch.obs.shape[:-len(self.obs_space.shape)]
        actions = self.action_space.sample(key)
        actions = jnp.broadcast_to(actions, batch_shapes+actions.shape)
        return actions, PyTreeDict()

    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        return self.compute_actions(agent_state, sample_batch, key)


class RandomAgent(Agent):
    """
        An agent that takes random actions.
    """

    def init(self, key: chex.PRNGKey) -> AgentState:
        return AgentState(
            params={}
        )

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        batch_shapes = sample_batch.obs.shape[:-len(self.obs_space.shape)]

        action_sample_fn = self.action_space.sample
        for _ in range(len(batch_shapes)):
            action_sample_fn = jax.vmap(action_sample_fn)

        action_keys = jax.random.split(key, np.prod(
            batch_shapes)).reshape(*batch_shapes, 2)

        actions = action_sample_fn(action_keys)
        return actions, PyTreeDict()

    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        return self.compute_actions(agent_state, sample_batch, key)
