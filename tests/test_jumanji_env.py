"""Jumanji state conversion, structured observations and action spaces."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jumanji = pytest.importorskip("jumanji")

from evorl.envs import Discrete, SpaceContainer
from evorl.envs.jumanji import (
    JumanjiAdapter,
    create_jumanji_env,
)
from evorl.types import PyTreeDict
from tests.helpers.jumanji import CounterEnv


def test_step_matches_native_state_and_preserves_input(jit):
    native = jumanji.make("Maze-v0")
    env = JumanjiAdapter(native)
    reset = jax.jit(env.reset) if jit else env.reset
    step = jax.jit(env.step) if jit else env.step
    key = jax.random.key(0)
    native_state, _ = native.reset(key)
    state = reset(key)
    snapshot = jax.tree.map(lambda x: x, state)
    action = jnp.array(0)
    native_state, transition = native.step(native_state, action)
    next_state = step(state, action)
    assert jax.tree.all(
        jax.tree.map(jnp.array_equal, native_state, next_state.env_state)
    )
    assert jax.tree.all(jax.tree.map(jnp.array_equal, snapshot, state))
    np.testing.assert_array_equal(next_state.obs.walls, transition.observation.walls)
    np.testing.assert_array_equal(
        next_state.obs.agent_position.row, transition.observation.agent_position.row
    )
    np.testing.assert_array_equal(next_state.reward, transition.reward)
    np.testing.assert_array_equal(next_state.done, transition.last())
    np.testing.assert_array_equal(next_state.info.discount, transition.discount)
    assert isinstance(env.obs_space, SpaceContainer)
    assert isinstance(env.action_space, Discrete)
    assert env.obs_space.contains(next_state.obs)


def test_step_updates_extras_without_mutating_previous_state(jit):
    env = JumanjiAdapter(CounterEnv())
    state = env.reset(jax.random.key(0))
    step = jax.jit(env.step) if jit else env.step
    next_state = step(state, jnp.array(0))
    assert isinstance(state.info, PyTreeDict)
    assert state.info.counter == 0
    assert next_state.info.counter == 1
    assert state.info is not next_state.info


def test_multicvrp_actions_match_available_nodes():
    env = create_jumanji_env("MultiCVRP-v0")
    node_count = env.unwrapped.observation_spec.action_mask.shape[-1]
    state = jax.jit(env.reset)(jax.random.key(1))
    assert state.env_state.nodes.coordinates.shape[0] == node_count
    for space in env.action_space.values():
        assert space.n == node_count
        assert space.contains(node_count - 1)
        assert not space.contains(node_count)
        samples = jax.jit(jax.vmap(space.sample))(
            jax.random.split(jax.random.key(0), 128)
        )
        assert jnp.all(samples < node_count)
