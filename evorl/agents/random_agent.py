from collections.abc import Callable
import chex
import jax
import jax.numpy as jnp
import numpy as np

from evorl.sample_batch import SampleBatch
from evorl.types import Action, PolicyExtraInfo, PyTreeDict, pytree_field
from evorl.envs import Space

from .agent import Agent, AgentState

EMPTY_RANDOM_AGENT_STATE = AgentState(params={})


class DebugRandomAgent(Agent):
    """
    An agent that takes random actions.
    Used for testing and debugging.
    """

    obs_space_shape: tuple[int] = pytree_field(lazy_init=True)
    action_sample_fn: Callable[[chex.PRNGKey], Action] = pytree_field(lazy_init=True)

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        self.set_frozen_attr("obs_space_shape", obs_space.shape)
        self.set_frozen_attr("action_sample_fn", action_space.sample)
        return EMPTY_RANDOM_AGENT_STATE

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        batch_shapes = sample_batch.obs.shape[: -len(self.obs_space_shape)]
        actions = self.action_sample_fn(key)
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

    obs_space_shape: tuple[int] = pytree_field(lazy_init=True)
    action_sample_fn: callable = pytree_field(lazy_init=True)

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        self.set_frozen_attr("obs_space_shape", obs_space.shape)
        self.set_frozen_attr("action_sample_fn", action_space.sample)
        return EMPTY_RANDOM_AGENT_STATE

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        batch_shapes = sample_batch.obs.shape[: -len(self.obs_space_shape)]

        action_sample_fn = self.action_sample_fn
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
