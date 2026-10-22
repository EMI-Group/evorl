import logging
from typing import Any

import jax
from flax import nnx
import jax.numpy as jnp

from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from omegaconf import DictConfig

from evorl.networks import make_policy_network
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action,
    PolicyExtraInfo,
    PyTreeDict,
    pytree_field,
)
from evorl.utils import running_statistics
from evorl.envs import Discrete, Space, Box

from evorl.agent import Agent, AgentState

logger = logging.getLogger(__name__)


class StochasticECAgent(Agent):
    """Stochastic Agent.

    Support continuous action space in [-1, 1] via TanhNormal distribution or
    discrete action space via Softmax distribution. Spectral-normalization
    vectors are stored alongside weights in `model_state`; action evaluation uses local
    model copies and does not persist their power-iteration updates.
    """

    continuous_action: bool
    agent_network_cfg: DictConfig = pytree_field(static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        if not isinstance(action_space, (Box, Discrete)):
            raise TypeError(f"Unsupported action space: {action_space}")
        cfg = self.agent_network_cfg
        obs_key = cfg.get("policy_obs_key", "")
        policy = make_policy_network(
            input_size=(obs_space.spaces[obs_key] if obs_key else obs_space).shape[-1],
            action_size=action_space.shape[0] * 2
            if self.continuous_action
            else action_space.n,
            hidden_layer_sizes=cfg.actor_hidden_layer_sizes,
            norm_layer_type=cfg.norm_layer_type,
            use_bias=cfg.use_bias,
            obs_key=obs_key,
            rngs=nnx.Rngs(key),
        )

        if self.normalize_obs:
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(
                obs_space.sample(key)
            )
        else:
            obs_preprocessor_state = None

        return AgentState(
            model_state=nnx.unpack(nnx.Dict(policy=policy)),
            obs_preprocessor_state=obs_preprocessor_state,
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = nnx.merge(agent_state.model_state, copy=True).policy(obs)

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
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = nnx.merge(agent_state.model_state, copy=True).policy(obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.mode()

        return actions, PyTreeDict()


class DeterministicECAgent(Agent):
    """Deterministic agent for continuous actions in [-1, 1].

    Spectral-normalization vectors stay outside the evolutionary parameters.
    Local power-iteration updates during inference are not persisted.
    """

    agent_network_cfg: DictConfig = pytree_field(static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        assert isinstance(action_space, Box), (
            "Only continuous action space is supported."
        )
        cfg = self.agent_network_cfg
        obs_key = cfg.get("policy_obs_key", "")
        policy = make_policy_network(
            input_size=(obs_space.spaces[obs_key] if obs_key else obs_space).shape[-1],
            action_size=action_space.shape[0],
            hidden_layer_sizes=cfg.actor_hidden_layer_sizes,
            norm_layer_type=cfg.norm_layer_type,
            use_bias=cfg.use_bias,
            obs_key=obs_key,
            activation_final=jnp.tanh,
            rngs=nnx.Rngs(key),
        )

        if self.normalize_obs:
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(
                obs_space.sample(key)
            )
        else:
            obs_preprocessor_state = None

        return AgentState(
            model_state=nnx.unpack(nnx.Dict(policy=policy)),
            obs_preprocessor_state=obs_preprocessor_state,
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        actions = nnx.merge(agent_state.model_state, copy=True).policy(obs)

        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        return self.compute_actions(agent_state, sample_batch, key)
