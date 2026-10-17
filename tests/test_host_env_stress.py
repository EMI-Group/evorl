"""Opt-in large host batches; use --host-env-stress to allocate native workers."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from evorl.envs import create_env, AutoresetMode

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "host_envs,num_envs",
    [("envpool", 4096), ("envpool", 16384), ("envpool", 65536), ("gymnasium", 16)],
    indirect=["host_envs"],
)
def test_large_nested_host_batch(host_envs, num_envs, request):
    if not request.config.getoption("--host-env-stress"):
        pytest.skip("Large native batches require --host-env-stress")
    backend, environments = host_envs
    kwargs = dict(num_threads=8) if backend == "envpool" else {}
    env = create_env(
        OmegaConf.create(dict(env_type=backend, env_name="CartPole-v1")),
        parallel=num_envs // 4,
        episode_length=16,
        autoreset_mode=AutoresetMode.ENVPOOL,
        **kwargs,
    )

    def run(key):
        state = env.reset(key)

        def body(state, _):
            state = env.step(state, jnp.zeros(env.num_envs, jnp.int32))
            return state, state.reward.sum()

        return jax.lax.scan(body, state, None, length=64)

    run = jax.jit(jax.vmap(jax.vmap(run)))
    keys = jax.random.split(jax.random.key(0), (2, 2))
    start = time.perf_counter()
    state, rewards = run(keys)
    jax.block_until_ready(state)
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    state, rewards = run(keys)
    jax.block_until_ready(state)
    seconds = time.perf_counter() - start
    assert environments[-1].num_envs == num_envs
    assert state.obs.shape == (2, 2, num_envs // 4, 4)
    assert rewards.shape == (2, 2, 64)
    assert np.isfinite(state.obs).all()
    assert np.isfinite(rewards).all()
    np.testing.assert_array_equal(state._internal.callback_token, 64)
    print(
        f"{backend}: {num_envs} environments on {jax.default_backend()}, "
        f"compile+64 steps={first_seconds:.3f}s, repeat={seconds:.3f}s"
    )
