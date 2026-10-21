"""Check Gymnax construction and parameter overrides with the real backend."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("gymnax")

from evorl.envs import create_env
from evorl.envs.gymnax import create_gymnax_env, create_wrapped_gymnax_env
from evorl.envs.wrappers.training_wrapper import AutoresetMode

CONFIGS = sorted(
    (Path(__file__).resolve().parents[1] / "configs/env/gymnax").glob("*.yaml")
)


@pytest.mark.parametrize("factory", ["base", "wrapped", "config", "override"])
def test_gymnax_constructor_and_params(factory):
    kwargs = dict(
        env_kwargs={"size": 4},
        deterministic=False,
        max_steps_in_episode=1,
    )
    if factory == "base":
        env = create_gymnax_env("DeepSea-bsuite", **kwargs)
        batch_shape = ()
    else:
        if factory == "wrapped":
            env = create_wrapped_gymnax_env(
                "DeepSea-bsuite", parallel=2, episode_length=2, **kwargs
            )
        else:
            config = OmegaConf.create(
                dict(
                    env_name="DeepSea-bsuite",
                    env_type="gymnax",
                    env_kwargs=dict(size=4 if factory == "config" else 8),
                )
            )
            if factory == "config":
                kwargs.pop("env_kwargs")
            env = create_env(config, parallel=2, episode_length=2, **kwargs)
        batch_shape = (2,)

    assert env.obs_space.shape == (16,)
    state = jax.jit(env.reset)(jax.random.key(0))
    assert state.obs.shape == (*batch_shape, 16)
    assert state.env_state.action_mapping.shape == (*batch_shape, 4, 4)
    np.testing.assert_array_equal(state.info.env_params.deterministic, False)
    np.testing.assert_array_equal(state.info.env_params.max_steps_in_episode, 1)

    action = jnp.zeros(batch_shape, dtype=jnp.int32)
    step = jax.jit(env.step)
    state = step(state, action)
    np.testing.assert_array_equal(state.done, np.zeros(batch_shape))
    state = step(state, action)
    if factory == "base":
        np.testing.assert_array_equal(state.done, np.zeros(batch_shape))
    else:
        np.testing.assert_array_equal(state.done, np.ones(batch_shape))
        np.testing.assert_array_equal(state.info.termination, np.zeros(batch_shape))
        np.testing.assert_array_equal(state.info.truncation, np.ones(batch_shape))


@pytest.mark.parametrize("create", [create_gymnax_env, create_wrapped_gymnax_env])
@pytest.mark.parametrize(
    "kwargs,field",
    [
        ({"env_kwargs": {"szie": 4}}, "szie"),
        ({"determinitsic": False}, "determinitsic"),
        ({"size": 4}, "size"),
    ],
    ids=[
        "unknown-constructor-argument",
        "unknown-param",
        "misplaced-constructor-argument",
    ],
)
def test_gymnax_unknown_arguments(create, kwargs, field):
    with pytest.raises(TypeError, match=field):
        create("DeepSea-bsuite", **kwargs)


@pytest.mark.integration
@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.stem)
def test_configured_environment_reset_and_step(config_path):
    config = OmegaConf.load(config_path)
    env = create_env(config, parallel=2, record_ori_obs=True)
    state = jax.jit(env.reset)(jax.random.key(0))
    actions = jax.vmap(env.action_space.sample)(jax.random.split(jax.random.key(1), 2))
    state = jax.jit(env.step)(state, actions)
    assert state.obs.shape == (2, *env.obs_space.shape)
    assert state.reward.shape == state.done.shape == (2,)
    assert state.reward.dtype == state.done.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(state.obs))
    assert jnp.all(jnp.isfinite(state.reward))


@pytest.mark.integration
@pytest.mark.parametrize("mode", list(AutoresetMode))
@pytest.mark.parametrize("env_name", ["CartPole-v1", "Pendulum-v1", "Breakout-MinAtar"])
def test_nested_vmap_matches_independent_worlds(env_name, mode):
    env = create_wrapped_gymnax_env(
        env_name,
        parallel=2,
        episode_length=2,
        autoreset_mode=mode,
        record_ori_obs=True,
        discount=0.5,
    )
    keys = jax.random.split(jax.random.key(0), (2, 1))
    state = jax.jit(jax.vmap(jax.vmap(env.reset)))(keys)
    reset = jax.jit(env.reset)
    serial = [reset(keys[i, 0]) for i in range(2)]
    step = jax.jit(env.step)
    nested_step = jax.jit(jax.vmap(jax.vmap(env.step)))
    actions = jax.vmap(jax.vmap(jax.vmap(env.action_space.sample)))(
        jax.random.split(jax.random.key(1), (2, 1, 2))
    )
    for _ in range(4):
        state = nested_step(state, actions)
        serial = [step(serial[i], actions[i, 0]) for i in range(2)]
        expected = jax.tree.map(lambda *xs: jnp.stack(xs)[:, None], *serial)
        assert jax.tree.structure(state) == jax.tree.structure(expected)
        for actual, reference in zip(
            jax.tree.leaves(state), jax.tree.leaves(expected), strict=True
        ):
            if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(
                    jax.random.key_data(actual), jax.random.key_data(reference)
                )
                continue
            np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)
