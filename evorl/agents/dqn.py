import jax
import jax.numpy as jnp
from flax import struct


from .agent import Agent, AgentState
from evorl.networks import make_q_network
from evorl.utils import running_statistics

from typing import Dict, Tuple, Sequence
import optax
import chex
import distrax
import dataclasses

from evorl.types import (
    EnvLike, LossDict, Action, Params, PolicyExtraInfo, EnvState,
    Observation, SampleBatch
)


@struct.dataclass
class A2CNetworkParams:
    """Contains training state for the learner."""
    q_params: Params


@dataclasses.dataclass
class DQNAgent(Agent):
    """
        Double-DQN
    """
    q_hidden_layer_sizes: Tuple[int] = (256, 256)
    discount: float = 0.99
    eploration_epsilon: float = 0.1

    def init(self, key: chex.PRNGKey) -> AgentState:
        obs_size = self.obs_space.shape[0]
        action_size = self.action_space.n

        self.q_network, q_init_fn = make_q_network(
            obs_size=obs_size,
            action_size=action_size,
            hidden_layer_sizes=self.q_hidden_layer_sizes,
            n_critics=1
        )

        key, q_key = jax.random.split(key)

        q_params = q_init_fn(q_key)

        return AgentState(
            params=A2CNetworkParams(q_params)
        )

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        """
            Args:
                sample_barch: [#env, ...]
        """

        qs = self.q_network.apply(
            agent_state.params.q_params, sample_batch.obs)

        #TODO: check EpsilonGreedy
        actions_dist = distrax.EpsilonGreedy(qs, epsilon=self.eploration_epsilon)
        actions = actions_dist.sample()

        return actions, dict(
            q_values=qs
        )
    
    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Action:
        """
            Args:
                sample_barch: [#env, ...]
        """
        qs = self.q_network.apply(
        agent_state.params.q_params, sample_batch.obs)

        actions_dist = distrax.EpsilonGreedy(qs, epsilon=self.eploration_epsilon)
        actions = actions_dist.mode()

        return actions
    
    def loss(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
        """
            Args:
                sample_barch: [B, ...]
        """
        qs = self.q_network.apply(
            agent_state.params.q_params, sample_batch.obs)

        
        td_error = None

        return dict(
            q_loss=td_error)
