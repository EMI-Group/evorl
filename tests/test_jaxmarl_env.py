"""JaxMARL native transitions, masks and EvoRL autoreset under nested vmap."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

jaxmarl = pytest.importorskip("jaxmarl")

from evorl.envs import MultiAgentEnv, SpaceContainer, create_env
from evorl.envs.jaxmarl import (
    create_jaxmarl_env,
    create_wrapped_jaxmarl_env,
    jaxmarl_space_to_evorl_space,
)
from evorl.envs.wrappers.training_wrapper import AutoresetMode
from evorl.types import PyTreeDict

CONFIGS = sorted(
    (Path(__file__).resolve().parents[1] / "configs/env/jaxmarl").rglob("*.yaml")
)
pytestmark = pytest.mark.integration


def assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    assert jax.tree.all(
        jax.tree.map(lambda x, y: jnp.allclose(x, y, atol=1e-6), actual, expected)
    )


def sample_actions(env, state):
    keys = jax.random.split(jax.random.key(5), len(env.agents))
    actions = {}
    for agent, key in zip(env.agents, keys, strict=True):
        if env._action_mask_fn is not None:
            actions[agent] = jnp.argmax(state.obs[agent].action_mask).astype(jnp.int32)
        else:
            actions[agent] = env.action_space[agent].sample(key)
    return actions


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.stem)
def test_configured_environment_matches_native_transition(config_path):
    config = OmegaConf.load(config_path)
    env = create_jaxmarl_env(config.env_name, **config.get("env_kwargs", {}))
    assert isinstance(env, MultiAgentEnv)
    key = jax.random.key(7)
    step_key, reset_key = jax.random.split(key)
    native_obs, native_state = jax.jit(env.env.reset)(reset_key)
    state = jax.jit(env.reset)(key)
    assert set(state.obs) == set(env.agents)
    assert set(state.reward) - {"__all__"} == set(env.agents)
    assert set(state.done) == {*env.agents, "__all__"}
    actions = sample_actions(env, state)
    _, action_key = jax.random.split(step_key)
    expected_obs, expected_state, expected_reward, expected_done, expected_info = (
        jax.jit(env.env.step_env)(action_key, native_state, actions)
    )
    result = jax.jit(env.step)(state, actions)
    assert_tree_equal(result.env_state, expected_state)
    for agent in env.agents:
        original_obs = state.obs[agent]
        next_obs = result.obs[agent]
        if env._action_mask_fn is not None:
            original_obs = original_obs.observation
            next_obs = next_obs.observation
            np.testing.assert_array_equal(
                result.obs[agent].action_mask,
                env._action_mask_fn(result.env_state)[agent].astype(bool),
            )
        np.testing.assert_allclose(original_obs, native_obs[agent], atol=1e-6)
        np.testing.assert_allclose(next_obs, expected_obs[agent], atol=1e-6)
        np.testing.assert_array_equal(result.reward[agent], expected_reward[agent])
        np.testing.assert_array_equal(result.done[agent], expected_done[agent])
        assert env.action_space[agent].contains(actions[agent])
        assert env.obs_space[agent].contains(state.obs[agent])
        assert env.obs_space[agent].contains(result.obs[agent])
    np.testing.assert_array_equal(result.done["__all__"], expected_done["__all__"])
    if "__all__" in expected_reward:
        np.testing.assert_array_equal(
            result.reward["__all__"], expected_reward["__all__"]
        )
    for name, value in expected_info.items():
        assert_tree_equal(result.info[name], PyTreeDict._nested_convert(value))
    if "world_state" in native_obs:
        np.testing.assert_allclose(state.info.global_obs, native_obs["world_state"])
        np.testing.assert_allclose(result.info.global_obs, expected_obs["world_state"])


@pytest.mark.parametrize("mode", list(AutoresetMode))
@pytest.mark.parametrize(
    "env_name",
    [
        "MPE_simple_spread_v3",
        "HeuristicEnemySMAX",
        "hanabi",
        "overcooked_v2",
        "coin_game",
        "jaxnav",
    ],
)
def test_nested_vmap_matches_independent_worlds(env_name, mode):
    env = create_wrapped_jaxmarl_env(
        env_name,
        episode_length=2,
        parallel=2,
        autoreset_mode=mode,
        record_ori_obs=True,
        discount=0.5,
    )
    keys = jax.random.split(jax.random.key(9), (2, 1))
    reset = jax.jit(jax.vmap(jax.vmap(env.reset)))
    step = jax.jit(jax.vmap(jax.vmap(env.step)))
    state = reset(keys)
    serial = [jax.jit(env.reset)(keys[i, 0]) for i in range(2)]
    native = env.unwrapped
    actions = {
        agent: jnp.zeros(
            (2, 1, 2, *space.shape),
            dtype=(jnp.int32 if not hasattr(space, "low") else space.low.dtype),
        )
        for agent, space in env.action_space.items()
    }
    # Hanabi receives only currently legal actions, including the inactive
    # players' no-op; all agents still provide an action each timestep.
    for step_index in range(4):
        if native._action_mask_fn is not None:
            actions = {
                agent: jnp.argmax(state.obs[agent].action_mask, axis=-1).astype(
                    jnp.int32
                )
                for agent in native.agents
            }
        terminal_obs = None
        if (
            native._action_mask_fn is not None
            and mode != AutoresetMode.ENVPOOL
            and step_index == 1
        ):
            terminal_obs = jax.jit(jax.vmap(jax.vmap(jax.vmap(native.step))))(
                state, actions
            ).obs
        state = step(state, actions)
        if terminal_obs is not None:
            assert_tree_equal(state.info.ori_obs, terminal_obs)
        next_serial = []
        for i in range(2):
            scalar = serial[i]
            scalar_actions = jax.tree.map(lambda x, i=i: x[i, 0], actions)
            scalar = jax.jit(env.step)(scalar, scalar_actions)
            next_serial.append(scalar)
            assert_tree_equal(jax.tree.map(lambda x, i=i: x[i, 0], state), scalar)
        serial = next_serial
        if native._action_mask_fn is not None:
            masks = jax.vmap(jax.vmap(jax.vmap(native._action_mask_fn)))(
                state.env_state
            )
            for agent in native.agents:
                np.testing.assert_array_equal(
                    state.obs[agent].action_mask, masks[agent].astype(bool)
                )
        if "global_obs" in state.info:
            global_obs = jax.vmap(jax.vmap(jax.vmap(native.env.get_world_state)))(
                state.env_state
            )
            assert_tree_equal(state.info.global_obs, global_obs)
    assert state.done["__all__"].shape == (2, 1, 2)


@pytest.mark.parametrize(
    "overrides",
    [{}, {"env_kwargs": {"layout": "cramped_room"}}, {"layout": "cramped_room"}],
)
def test_factory_forwards_native_configuration(overrides):
    config = PyTreeDict(
        env_type="jaxmarl",
        env_name="overcooked_v2",
        env_kwargs=PyTreeDict(layout="forced_coord"),
    )
    env = create_env(
        config, parallel=2, episode_length=2, record_ori_obs=True, **overrides
    )
    state = jax.jit(env.reset)(jax.random.key(0))
    assert state.obs.agent_0.shape[0] == 2
    assert env.unwrapped.env.width == 5
    assert env.unwrapped.env.height == (4 if overrides else 5)


def test_space_conversion_preserves_nested_structure():
    from jaxmarl.environments import spaces

    native = spaces.Dict(
        {
            "vector": spaces.Box(-1, 1, (2,)),
            "choice": spaces.MultiDiscrete([2, 3]),
            "pair": spaces.Tuple((spaces.Discrete(4), spaces.Box(0, 1, ()))),
        }
    )
    space = jaxmarl_space_to_evorl_space(native)
    assert isinstance(space, SpaceContainer)
    assert space.contains(space.sample(jax.random.key(0)))


@pytest.mark.parametrize(
    "env_name", ["hopper_3x1", "storm", "JaxRobotarium_navigation"]
)
def test_unadapted_environment_is_rejected(env_name):
    with pytest.raises(ValueError, match="Unsupported JaxMARL"):
        create_jaxmarl_env(env_name)
