import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.envs import (
    create_wrapped_brax_env,
    AutoresetMode,
    Box,
)
from evorl.envs.wrappers import get_wrapper
from evorl.envs.wrappers.training_wrapper import (
    EpisodeWrapper,
    OneEpisodeWrapper,
    VmapWrapper,
    VmapAutoResetWrapper,
    FastVmapAutoResetWrapper,
    VmapEnvPoolAutoResetWrapper,
)


def has_wrapper(env, wrapper_cls):
    return get_wrapper(env, wrapper_cls) is not None


def test_brax_env():
    num_envs = 7

    env = create_wrapped_brax_env("ant", parallel=num_envs)
    action_space = env.action_space
    obs_space = env.obs_space
    assert env.num_envs == num_envs
    assert isinstance(action_space, Box)
    assert isinstance(obs_space, Box)
    assert action_space.shape == (8,), action_space.shape
    assert obs_space.shape == (27,), obs_space.shape
    assert env.num_envs == num_envs


def test_has_brax_wrapper():
    num_envs = 3
    env = create_wrapped_brax_env(
        "ant", parallel=num_envs, autoreset_mode=AutoresetMode.NORMAL
    )

    assert has_wrapper(env, EpisodeWrapper)
    assert has_wrapper(env, VmapAutoResetWrapper)

    env = create_wrapped_brax_env(
        "ant", parallel=num_envs, autoreset_mode=AutoresetMode.FAST
    )
    assert has_wrapper(env, EpisodeWrapper)
    assert has_wrapper(env, FastVmapAutoResetWrapper)

    env = create_wrapped_brax_env(
        "ant", parallel=num_envs, autoreset_mode=AutoresetMode.ENVPOOL
    )
    assert has_wrapper(env, EpisodeWrapper)
    assert has_wrapper(env, VmapEnvPoolAutoResetWrapper)

    env = create_wrapped_brax_env(
        "ant", parallel=num_envs, autoreset_mode=AutoresetMode.DISABLED
    )
    assert has_wrapper(env, OneEpisodeWrapper)
    assert has_wrapper(env, VmapWrapper)


@pytest.mark.parametrize("autoreset_mode", list(AutoresetMode))
@pytest.mark.parametrize("obs_mode", ["ndarray", "dict_state"])
def test_brax_fast(autoreset_mode, obs_mode):
    env = create_wrapped_brax_env(
        "fast",
        parallel=2,
        episode_length=2,
        autoreset_mode=autoreset_mode,
        obs_mode=obs_mode,
        asymmetric_obs=obs_mode == "dict_state",
    )
    assert env.action_space.shape == (1,)
    np.testing.assert_array_equal(env.action_space.low, [-1.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0])
    expected_shapes = (
        {"state": (2, 2), "privileged_state": (2, 4)}
        if obs_mode == "dict_state"
        else (2, 2)
    )

    state = jax.jit(env.reset)(jax.random.key(0))
    assert jax.tree.map(lambda x: x.shape, state.obs) == expected_shapes
    step = jax.jit(env.step)
    for index in range(3):
        state = step(state, jnp.ones((2, 1)))
        assert jax.tree.map(lambda x: x.shape, state.obs) == expected_shapes
        assert jnp.all(jnp.isfinite(state.reward))
        done = index >= 1 if autoreset_mode == AutoresetMode.DISABLED else index == 1
        np.testing.assert_array_equal(state.done, np.full(2, done))
