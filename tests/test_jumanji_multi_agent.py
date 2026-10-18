"""Compare agent-keyed Jumanji transitions with their native joint transitions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jumanji = pytest.importorskip("jumanji")

from evorl.envs import Env, MultiAgentEnv, create_env
from evorl.envs.jumanji import create_jumanji_env
from evorl.types import PyTreeDict


MULTI_AGENT_TASKS = [
    "Cleaner-v0",
    "Connector-v3",
    "LevelBasedForaging-v0",
    "MMST-v0",
    "MultiCVRP-v0",
    "RobotWarehouse-v0",
    "SearchAndRescue-v0",
]


@pytest.mark.parametrize("env_name", MULTI_AGENT_TASKS)
@pytest.mark.integration
def test_multi_agent_step_preserves_native_rewards_actions_and_observations(env_name):
    env = create_jumanji_env(env_name)
    assert isinstance(env, MultiAgentEnv)
    assert not isinstance(env, Env)
    native = env.unwrapped
    key = jax.random.key(5)
    native_state, transition = jax.jit(native.reset)(key)
    state = jax.jit(env.reset)(key)
    action_spaces, obs_spaces = env.action_space, env.obs_space
    assert set(state.obs) == set(action_spaces) == set(obs_spaces) == set(env.agents)
    assert set(state.done) == {*env.agents, "__all__"}
    assert set(state.reward) - {"__all__"} == set(env.agents)
    joint_action = jnp.asarray(
        [
            space.sample(k)
            for space, k in zip(
                action_spaces.values(),
                jax.random.split(key, len(env.agents)),
                strict=True,
            )
        ],
        dtype=native.action_spec.dtype,
    )
    native.action_spec.validate(joint_action)
    # Reverse the dictionary order to catch adapters that rely on caller ordering.
    actions = {
        agent: joint_action[i] for i, agent in reversed(list(enumerate(env.agents)))
    }
    next_state = jax.jit(env.step)(state, actions)
    expected_state, expected = jax.jit(native.step)(native_state, joint_action)
    assert jax.tree.all(
        jax.tree.map(jnp.array_equal, next_state.env_state, expected_state)
    )
    for i, agent in enumerate(env.agents):
        assert action_spaces[agent].contains(actions[agent])
        assert obs_spaces[agent].contains(state.obs[agent])
        assert obs_spaces[agent].contains(next_state.obs[agent])
        native_reward = (
            expected.reward if expected.reward.ndim == 0 else expected.reward[i]
        )
        np.testing.assert_allclose(next_state.reward[agent], native_reward)
        native_discount = (
            expected.discount if expected.discount.ndim == 0 else expected.discount[i]
        )
        np.testing.assert_array_equal(next_state.info.discount[agent], native_discount)
        if hasattr(expected.observation, "action_mask"):
            np.testing.assert_array_equal(
                next_state.obs[agent].action_mask, expected.observation.action_mask[i]
            )
        assert state.obs[agent].agent_id == i
        assert state.done[agent] == 0
    if expected.reward.ndim == 0:
        np.testing.assert_allclose(next_state.reward["__all__"], expected.reward)
    else:
        assert "__all__" not in next_state.reward
    np.testing.assert_array_equal(next_state.done["__all__"], expected.last())
    # Global fields must not be accidentally sliced when their size equals N.
    if env_name == "MultiCVRP-v0":
        assert state.obs.agent_0.nodes.coordinates.shape == (21, 2)
        assert state.obs.agent_0.vehicles.coordinates.shape == (2,)
    elif env_name == "Connector-v3":
        np.testing.assert_array_equal(
            state.obs.agent_1.grid, transition.observation.grid
        )
    elif env_name == "Cleaner-v0":
        np.testing.assert_array_equal(
            state.obs.agent_1.agents_locations, transition.observation.agents_locations
        )
    elif env_name == "MMST-v0":
        np.testing.assert_array_equal(
            state.obs.agent_1.positions, transition.observation.positions
        )
    elif env_name == "SearchAndRescue-v0":
        np.testing.assert_array_equal(
            state.obs.agent_1.searcher_views, transition.observation.searcher_views[1]
        )
        assert state.obs.agent_1.positions.shape == (2,)


def test_factory_selects_multi_agent_api_without_reward_aggregation():
    env = create_env(
        PyTreeDict(env_type="jumanji", env_name="LevelBasedForaging-v0"), parallel=2
    )
    assert isinstance(env.unwrapped, MultiAgentEnv)
    state = jax.jit(env.reset)(jax.random.key(0))
    assert set(state.reward) == set(env.agents)
    for reward in state.reward.values():
        assert reward.shape == (2,)


def test_native_individual_termination_does_not_end_episode(monkeypatch):
    from jumanji.types import StepType

    env = create_jumanji_env("Connector-v3")
    native_step = env.unwrapped.step

    def step(state, action):
        state, timestep = native_step(state, action)
        return state, timestep.replace(
            step_type=StepType.MID,
            discount=jnp.ones(len(env.agents)).at[0].set(0),
        )

    monkeypatch.setattr(env.unwrapped, "step", step)
    initial = jax.jit(env.reset)(jax.random.key(0))
    result = jax.jit(env.step)(initial, {agent: jnp.array(0) for agent in env.agents})
    assert result.done.agent_0 == result._internal.termination.agent_0 == 1
    assert result.done.agent_1 == result.done["__all__"] == 0
    assert result._internal.truncation.agent_0 == 0
    assert result._internal.termination["__all__"] == 0


@pytest.mark.parametrize("actions", [{}, {"unexpected": jnp.array(0)}])
def test_incomplete_or_unknown_agent_actions_are_rejected(actions):
    env = create_jumanji_env("LevelBasedForaging-v0")
    state = jax.jit(env.reset)(jax.random.key(0))
    with pytest.raises(ValueError, match="exactly one entry per agent"):
        env.step(state, actions)
