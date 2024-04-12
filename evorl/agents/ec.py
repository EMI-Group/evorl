import jax
import jax.numpy as jnp
from flax import struct
import math

from omegaconf import DictConfig


from evorl.sample_batch import SampleBatch
from evorl.networks import make_policy_network, make_value_network
from evorl.utils import running_statistics
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.toolkits import (
    compute_gae, flatten_rollout_trajectory,
    average_episode_discount_return
)
from evorl.workflows import OnPolicyRLWorkflow
from evorl.agents import AgentState
from evorl.distributed import agent_gradient_update, tree_unpmap, psum
from evorl.envs import create_env, Env, EnvState
from evorl.evaluator import Evaluator
from .agent import Agent, AgentState

from evox import State
# from evorl.types import State


import orbax.checkpoint as ocp
import chex
import optax
from evorl.types import (
    LossDict, Action, Params, PolicyExtraInfo, PyTreeDict, pytree_field,
    MISSING_REWARD
)
from evorl.metrics import TrainMetric, WorkflowMetric
from typing import Tuple, Sequence, Optional, Any
import logging
import flax.linen as nn
from flax import struct

logger = logging.getLogger(__name__)


@struct.dataclass
class ECNetworkParams:
    """Contains training state for the learner."""
    policy_params: Params


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

    def loss(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
        raise NotImplementedError()