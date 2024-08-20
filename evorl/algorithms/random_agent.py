from collections.abc import Callable
import chex
import jax
import jax.numpy as jnp
import numpy as np

from evorl.sample_batch import SampleBatch
from evorl.types import Action, PolicyExtraInfo, PyTreeDict, pytree_field
from evorl.envs import Space

from .agent import Agent, AgentState


class DebugRandomAgent(Agent):
    """
    An agent that takes random actions.
    Used for testing and debugging.
    """

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        return AgentState(params={}, obs_space=obs_space, action_space=action_space)

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        batch_shapes = sample_batch.obs.shape[: -len(agent_state.obs_space.shape)]
        actions = agent_state.action_space.sample(key)
        actions = jnp.broadcast_to(actions, batch_shapes + actions.shape)
        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        return self.compute_actions(agent_state, sample_batch, key)


class RandomAgent(Agent):
    """
    An agent that takes random actions.
    """

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        return AgentState(params={}, obs_space=obs_space, action_space=action_space)

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        batch_shapes = sample_batch.obs.shape[: -len(agent_state.obs_space.shape)]

        action_sample_fn = agent_state.action_space.sample
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
