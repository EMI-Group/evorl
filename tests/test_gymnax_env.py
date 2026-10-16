"""Check Gymnax construction and parameter overrides with the real backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("gymnax")

from evorl.envs import create_env
from evorl.envs.gymnax import create_gymnax_env, create_wrapped_gymnax_env


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
