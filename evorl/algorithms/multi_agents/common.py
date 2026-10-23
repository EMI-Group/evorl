"""Observation layouts and masked reductions shared by MARL algorithms."""

import math
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from evorl.envs import Box, Discrete
from evorl.agent import Agent
from evorl.types import PyTreeDict, pytree_field
from evorl.utils import running_statistics


class MultiAgentAgent(Agent):
    """Static configuration for agent-keyed networks and heterogeneous spaces."""

    agents: tuple = pytree_field(static=True)
    observation_shapes: tuple = pytree_field(static=True)
    global_observation_shapes: tuple | None = pytree_field(static=True)
    action_shapes: tuple = pytree_field(static=True)
    action_sizes: tuple = pytree_field(static=True)
    continuous_actions: tuple = pytree_field(static=True)
    action_bounds: tuple = pytree_field(static=True)
    agent_network_cfg: DictConfig = pytree_field(static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)


def init_preprocessing(agent):
    """Initialize feature statistics without including legal-action masks."""
    if agent.obs_preprocessor is None:
        return None
    return PyTreeDict(
        local=PyTreeDict(
            {
                name: running_statistics.init_state(
                    jnp.zeros(sum(map(math.prod, shapes)))
                )
                for name, shapes in zip(
                    agent.agents, agent.observation_shapes, strict=True
                )
            }
        ),
        global_obs=(
            running_statistics.init_state(
                jnp.zeros(sum(map(math.prod, agent.global_observation_shapes)))
            )
            if agent.global_observation_shapes is not None
            else None
        ),
    )


def update_preprocessing(agent, agent_state, trajectory, axis_name):
    """Accumulate valid observations after the trajectory's targets are fixed."""
    if agent_state.obs_preprocessor_state is None:
        return agent_state
    previous = agent_state.obs_preprocessor_state
    local = PyTreeDict(
        {
            name: running_statistics.update(
                previous.local[name],
                flatten_observation(policy_observation(trajectory.obs[name]), shapes),
                weights=trajectory.extras.active[name],
                axis_name=axis_name,
            )
            for name, shapes in zip(agent.agents, agent.observation_shapes, strict=True)
        }
    )
    global_obs = previous.global_obs
    if agent.global_observation_shapes is not None:
        global_obs = running_statistics.update(
            global_obs,
            flatten_observation(
                trajectory.extras.global_obs, agent.global_observation_shapes
            ),
            axis_name=axis_name,
        )
    return agent_state.replace(
        obs_preprocessor_state=PyTreeDict(local=local, global_obs=global_obs)
    )


def policy_observation(obs):
    """Separate legal-action masks from the policy's observation features."""
    if isinstance(obs, Mapping) and "action_mask" in obs:
        return obs.get(
            "observation", {k: v for k, v in obs.items() if k != "action_mask"}
        )
    return obs


def observation_shapes(obs):
    """Return static feature shapes, preserving the pytree's leaf order."""
    return tuple(tuple(x.shape) for x in jax.tree.leaves(obs))


def flatten_observation(obs, shapes):
    """Flatten feature axes while retaining every leading batch axis."""
    leaves = []
    for value, shape in zip(jax.tree.leaves(obs), shapes, strict=True):
        batch_shape = value.shape[: -len(shape)] if shape else value.shape
        leaves.append(
            value.astype(jnp.float32).reshape((*batch_shape, math.prod(shape)))
        )
    return jnp.concatenate(leaves, axis=-1)


def agent_layout(env):
    """Describe fixed agent IDs, observation features and supported actions.

    Returns:
        Keyword arguments for a static MARL agent. Structured numeric
        observations are flattened; action masks are kept out of the features.
        Actions are scalar `Discrete` or finite, floating-point `Box` spaces.
    """
    if not hasattr(env, "agents") or not isinstance(env.obs_space, Mapping):
        raise TypeError(
            "MARL workflows require a MultiAgentEnv with agent-keyed spaces."
        )
    agents = tuple(env.agents)
    if (
        not agents
        or set(env.action_space) != set(agents)
        or set(env.obs_space) != set(agents)
    ):
        raise ValueError(
            "Observation and action spaces must match the fixed agent IDs."
        )
    shapes, action_shapes, action_sizes, continuous, bounds = [], [], [], [], []
    for name in agents:
        sample = env.obs_space[name].sample(jax.random.key(0))
        shapes.append(observation_shapes(policy_observation(sample)))
        space = env.action_space[name]
        if isinstance(space, Discrete):
            action_shapes.append(())
            action_sizes.append(space.n)
            continuous.append(False)
            bounds.append(((), ()))
        elif isinstance(space, Box) and jnp.issubdtype(space.low.dtype, jnp.floating):
            low, high = space.low.reshape(-1), space.high.reshape(-1)
            if not bool(jnp.all(jnp.isfinite(low) & jnp.isfinite(high) & (high > low))):
                raise ValueError(
                    "Continuous actions require finite, strictly ordered bounds."
                )
            action_shapes.append(tuple(space.shape))
            action_sizes.append(math.prod(space.shape))
            continuous.append(True)
            bounds.append((tuple(low.tolist()), tuple(high.tolist())))
        else:
            raise TypeError(f"Unsupported action space for {name}: {space}")
    abstract = jax.eval_shape(env.reset, jax.random.key(0))
    if any(abstract.reward[name].shape != (env.num_envs,) for name in agents):
        raise ValueError("MARL workflows require one scalar reward per agent.")
    global_shapes = None
    if "global_obs" in abstract.info:
        global_shapes = tuple(
            tuple(x.shape[1:]) for x in jax.tree.leaves(abstract.info.global_obs)
        )
    return dict(
        agents=agents,
        observation_shapes=tuple(shapes),
        global_observation_shapes=global_shapes,
        action_shapes=tuple(action_shapes),
        action_sizes=tuple(action_sizes),
        continuous_actions=tuple(continuous),
        action_bounds=tuple(bounds),
    )


def local_features(agent, agent_state, observations):
    """Extract each agent's local input using the saved preprocessing state."""
    features = PyTreeDict()
    for name, shapes in zip(agent.agents, agent.observation_shapes, strict=True):
        value = flatten_observation(policy_observation(observations[name]), shapes)
        if agent_state.obs_preprocessor_state is not None:
            value = agent.obs_preprocessor(
                value, agent_state.obs_preprocessor_state.local[name]
            )
        features[name] = value
    return features


def centralized_features(agent, agent_state, observations, global_obs):
    """Use native global observations, or concatenate all local observations."""
    if agent.global_observation_shapes is None:
        features = local_features(agent, agent_state, observations)
        return jnp.concatenate([features[name] for name in agent.agents], axis=-1)
    value = flatten_observation(global_obs, agent.global_observation_shapes)
    if agent_state.obs_preprocessor_state is not None:
        value = agent.obs_preprocessor(
            value, agent_state.obs_preprocessor_state.global_obs
        )
    return value


def masked_logits(logits, obs):
    """Apply legal actions, using action zero when an inactive agent has none."""
    if not isinstance(obs, Mapping) or "action_mask" not in obs:
        return logits
    mask = obs["action_mask"].astype(jnp.bool_)
    fallback = jnp.arange(logits.shape[-1]) == 0
    mask = jnp.where(mask.any(axis=-1, keepdims=True), mask, fallback)
    return jnp.where(mask, logits, jnp.finfo(logits.dtype).min)


def masked_mean(value, mask, axis_name=None):
    """Weight local losses so an outer `pmean` gives the global valid mean."""
    count = mask.sum()
    world_size = 1
    if axis_name is not None:
        count = jax.lax.psum(count, axis_name)
        world_size = jax.lax.psum(jnp.float32(1), axis_name)
    return jnp.where(mask, value, 0).sum() * world_size / jnp.maximum(count, 1)


def scale_action(action, bounds, shape):
    """Map normalized actions into the environment's Box bounds."""
    low, high = map(jnp.asarray, bounds)
    value = low + (action + 1) * (high - low) / 2
    return value.reshape((*action.shape[:-1], *shape))


def normalize_action(action, bounds, shape):
    """Map a Box action back to a flat vector in `[-1, 1]`."""
    batch_shape = action.shape[: -len(shape)] if shape else action.shape
    low, high = map(jnp.asarray, bounds)
    return 2 * (action.reshape((*batch_shape, -1)) - low) / (high - low) - 1
