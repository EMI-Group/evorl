import logging
from typing import Any

import chex
import flax.linen as nn
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.networks import make_policy_network
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action,
    Params,
    PolicyExtraInfo,
    PyTreeDict,
    pytree_field,
    PyTreeData,
)
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_get
from evorl.envs import Space, Box, Discrete

from evorl.agent import Agent, AgentState

logger = logging.getLogger(__name__)


class ECNetworkParams(PyTreeData):
    """Contains training state for the learner."""

    policy_params: Params


class StochasticECAgent(Agent):
    """Stochastic Agent.

    Support continuous action space in [-1, 1] via TanhNormal distribution or discrete action space via Softmax distribution.
    """

    continuous_action: bool
    policy_network: nn.Module
    obs_preprocessor: Any = pytree_field(default=None, static=True)

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        dummy_obs = jtu.tree_map(lambda x: x[None, ...], obs_space.sample(key))
        policy_params = self.policy_network.init(key, dummy_obs)

        params_state = ECNetworkParams(
            policy_params=policy_params,
        )

        if self.normalize_obs:
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(
                tree_get(dummy_obs, 0)
            )
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state, obs_preprocessor_state=obs_preprocessor_state
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.sample(seed=key)

        policy_extras = PyTreeDict(
            # raw_action=raw_actions,
            # logp=actions_dist.log_prob(actions)
        )

        return actions, policy_extras

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.mode()

        return actions, PyTreeDict()


class DeterministicECAgent(Agent):
    """Deterministic Agent for continuous action space in [-1, 1]."""

    policy_network: nn.Module
    obs_preprocessor: Any = pytree_field(default=None, static=True)

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        dummy_obs = jtu.tree_map(lambda x: x[None, ...], obs_space.sample(key))
        policy_params = self.policy_network.init(key, dummy_obs)

        params_state = ECNetworkParams(
            policy_params=policy_params,
        )

        if self.normalize_obs:
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(
                tree_get(dummy_obs, 0)
            )
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state, obs_preprocessor_state=obs_preprocessor_state
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        actions = self.policy_network.apply(agent_state.params.policy_params, obs)

        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        return self.compute_actions(agent_state, sample_batch, key)


def make_stochastic_ec_agent(
    action_space: Space,
    actor_hidden_layer_sizes: tuple[int] = (256, 256),
    use_bias: bool = True,
    norm_layer_type: str = "none",
    normalize_obs: bool = False,
    policy_obs_key: str = "",
):
    if isinstance(action_space, Box):
        action_size = action_space.shape[0] * 2
        continuous_action = True
    elif isinstance(action_space, Discrete):
        action_size = action_space.n
        continuous_action = False
    else:
        raise NotImplementedError(f"Unsupported action space: {action_space}")

    policy_network = make_policy_network(
        action_size=action_size,
        hidden_layer_sizes=actor_hidden_layer_sizes,
        norm_layer_type=norm_layer_type,
        use_bias=use_bias,
        obs_key=policy_obs_key,
    )

    if normalize_obs:
        obs_preprocessor = running_statistics.normalize
    else:
        obs_preprocessor = None

    return StochasticECAgent(
        continuous_action=continuous_action,
        policy_network=policy_network,
        obs_preprocessor=obs_preprocessor,
    )


def make_deterministic_ec_agent(
    action_space: Space,
    actor_hidden_layer_sizes: tuple[int] = (256, 256),
    use_bias: bool = True,
    norm_layer_type: str = "none",
    normalize_obs: bool = False,
    policy_obs_key: str = "",
):
    assert isinstance(action_space, Box), "Only continue action space is supported."

    action_size = action_space.shape[0]

    policy_network = make_policy_network(
        action_size=action_size,
        hidden_layer_sizes=actor_hidden_layer_sizes,
        use_bias=use_bias,
        activation_final=nn.tanh,
        norm_layer_type=norm_layer_type,
        obs_key=policy_obs_key,
    )

    if normalize_obs:
        obs_preprocessor = running_statistics.normalize
    else:
        obs_preprocessor = None

    return DeterministicECAgent(
        policy_network=policy_network,
        obs_preprocessor=obs_preprocessor,
    )
