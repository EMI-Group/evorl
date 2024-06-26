import jax
import jax.numpy as jnp
from flax import struct

from evorl.sample_batch import SampleBatch
from evorl.networks.linear import ActivationFn, MLP
from evorl.utils import running_statistics
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist

from evorl.agents import AgentState
from ..agent import Agent, AgentState


import chex
from evorl.types import (
    LossDict, Action, Params, PolicyExtraInfo, PyTreeDict, pytree_field,
)

from typing import Tuple, Sequence, Optional, Any
import logging
import flax.linen as nn
from flax import struct

logger = logging.getLogger(__name__)


@struct.dataclass
class ECNetworkParams:
    """Contains training state for the learner."""
    policy_params: Params


def make_policy_network(
        action_size: int,
        obs_size: int,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = nn.relu) -> nn.Module:
    """Creates a policy network w/ LayerNorm."""
    policy_model = MLP(
        layer_sizes=list(hidden_layer_sizes) + [action_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        norm_layer=nn.LayerNorm)

    def init_fn(rng): return policy_model.init(rng, jnp.zeros((1, obs_size)))

    return policy_model, init_fn


class StochasticECAgent(Agent):
    actor_hidden_layer_sizes: Tuple[int] = (256, 256)
    normalize_obs: bool = False
    continuous_action: bool = False
    policy_network: nn.Module = pytree_field(lazy_init=True)  # nn.Module is ok
    obs_preprocessor: Any = pytree_field(lazy_init=True, pytree_node=False)

    def init(self, key: chex.PRNGKey) -> AgentState:

        obs_size = self.obs_space.shape[0]

        if self.continuous_action:
            action_size = self.action_space.shape[0]
            action_size *= 2
        else:
            action_size = self.action_space.n

        policy_key, value_key, obs_preprocessor_key = jax.random.split(key, 3)
        policy_network, policy_init_fn = make_policy_network(
            action_size=action_size,
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes
        )
        policy_params = policy_init_fn(policy_key)

        self.set_frozen_attr('policy_network', policy_network)

        params_state = ECNetworkParams(
            policy_params=policy_params,
        )

        if self.normalize_obs:
            obs_preprocessor = running_statistics.normalize
            self.set_frozen_attr('obs_preprocessor', obs_preprocessor)
            dummy_obs = self.obs_space.sample(obs_preprocessor_key)
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state
        )

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        """
            Args:
                sample_barch: [#env, ...]
        """
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(
            agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(
                *jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.sample(seed=key)

        policy_extras = PyTreeDict(
            # raw_action=raw_actions,
            # logp=actions_dist.log_prob(actions)
        )

        return jax.lax.stop_gradient(actions), policy_extras

    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        """
            Args:
                sample_barch: [#env, ...]
        """
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(
            agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(
                *jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.mode()

        return jax.lax.stop_gradient(actions), PyTreeDict()


class DeterministicECAgent(Agent):
    actor_hidden_layer_sizes: Tuple[int] = (256, 256)
    normalize_obs: bool = False
    policy_network: nn.Module = pytree_field(lazy_init=True)  # nn.Module is ok
    obs_preprocessor: Any = pytree_field(lazy_init=True, pytree_node=False)

    def init(self, key: chex.PRNGKey) -> AgentState:

        obs_size = self.obs_space.shape[0]

        # it must be continuous action
        action_size = self.action_space.shape[0]

        policy_key, obs_preprocessor_key = jax.random.split(key, 2)
        policy_network, policy_init_fn = make_policy_network(
            action_size=action_size,
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes
        )
        policy_params = policy_init_fn(policy_key)

        self.set_frozen_attr('policy_network', policy_network)

        params_state = ECNetworkParams(
            policy_params=policy_params,
        )

        if self.normalize_obs:
            obs_preprocessor = running_statistics.normalize
            self.set_frozen_attr('obs_preprocessor', obs_preprocessor)
            dummy_obs = self.obs_space.sample(obs_preprocessor_key)
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state
        )

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        """
            Args:
                sample_barch: [#env, ...]
        """
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(
            agent_state.params.policy_params, obs)

        actions = jnp.tanh(raw_actions)

        # Note: ECAgent always output best action w/o exploration noise

        return jax.lax.stop_gradient(actions), PyTreeDict()

    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        return self.compute_actions(agent_state, sample_batch, key)
