"""Check batch layouts against the optional host environment backends."""

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from evorl.envs import create_env
from evorl.envs.wrappers import AutoresetMode


@pytest.mark.parametrize("backend", ["gymnasium", "envpool"])
@pytest.mark.parametrize("env_name,obs_size", [("CartPole-v1", 4), ("Pendulum-v1", 3)])
@pytest.mark.parametrize("vmapped", [False, True], ids=["batch", "vmap"])
@pytest.mark.parametrize(
    "autoreset_mode",
    [AutoresetMode.ENVPOOL, AutoresetMode.DISABLED],
    ids=["autoreset", "one_episode"],
)
def test_host_env_action_batches(
    backend, env_name, obs_size, vmapped, autoreset_mode, monkeypatch
):
    library = pytest.importorskip(backend)
    module = importlib.import_module(f"evorl.envs.{backend}")
    factory_name = "make_vec" if backend == "gymnasium" else "make_gymnasium"
    original_factory = getattr(library, factory_name)
    environments = []

    def make_env(*args, **kwargs):
        env = original_factory(*args, **kwargs)
        environments.append(env)
        return env

    monkeypatch.setattr(library, factory_name, make_env)
    try:
        create = getattr(module, f"create_{backend}_env")
        kwargs = {"num_threads": 1} if backend == "envpool" else {}
        env = create(
            env_name,
            parallel=2,
            episode_length=2,
            autoreset_mode=autoreset_mode,
            **kwargs,
        )
        key = jax.random.key(0)
        reset, step = env.reset, env.step
        batch_shape = (2,)
        if vmapped:
            key = jax.random.split(key, 2)
            reset, step = jax.vmap(reset), jax.vmap(step)
            batch_shape = (2, 2)

        state = jax.jit(reset)(key)
        jax.block_until_ready(state)
        record_return = autoreset_mode != AutoresetMode.DISABLED
        assert ("episode_return" in state.info) == record_return
        action = jnp.zeros(
            (*batch_shape, *env.action_space.shape),
            dtype=jnp.int32 if env_name == "CartPole-v1" else jnp.float32,
        )
        step = jax.jit(step)
        for index in range(3):
            state = step(state, action)
            jax.block_until_ready(state)
            assert state.obs.shape == (*batch_shape, obs_size)
            assert state.reward.shape == batch_shape
            assert state.done.shape == batch_shape
            assert ("episode_return" in state.info) == record_return
            truncated = index == 1 if record_return else index >= 1
            np.testing.assert_array_equal(
                state.info.truncation, np.full(batch_shape, truncated)
            )
    finally:
        for env in environments:
            env.close()


def test_atari_configs():
    pytest.importorskip("ale_py")
    import gymnasium

    config_dir = Path(__file__).resolve().parents[1] / "configs/env/gymnasium/ALE"
    configs = sorted(config_dir.glob("*.yaml"))
    assert configs
    for path in configs:
        config = OmegaConf.load(path)
        assert config.env_name.startswith("ale_py:ALE/"), path.name
        assert config.obs_type == ("ram" if "-ram-" in path.name else "rgb"), path.name
        env_id = config.env_name.removeprefix("ale_py:")
        assert gymnasium.spec(env_id).id == env_id, path.name


@pytest.mark.parametrize(
    "config_name,override,obs_shape",
    [
        ("pong-v5", None, (210, 160, 3)),
        ("pong-ram-v5", None, (128,)),
        ("pong-v5", "ram", (128,)),
    ],
    ids=["rgb", "ram", "override"],
)
def test_atari_observations_from_config(config_name, override, obs_shape, monkeypatch):
    if importlib.util.find_spec("ale_py") is None:
        pytest.skip("Requires ale-py")
    import gymnasium

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/env/gymnasium/ALE"
        / f"{config_name}.yaml"
    )
    config = OmegaConf.load(config_path)
    original_factory = gymnasium.make_vec
    environments = []

    def make_env(*args, **kwargs):
        env = original_factory(*args, **kwargs)
        environments.append(env)
        return env

    monkeypatch.setattr(gymnasium, "make_vec", make_env)
    try:
        kwargs = {} if override is None else {"obs_type": override}
        env = create_env(config, parallel=2, **kwargs)
        assert env.obs_space.shape == obs_shape
        state = jax.jit(env.reset)(jax.random.key(0))
        assert state.obs.shape == (2, *obs_shape)
        assert state.obs.dtype == jnp.uint8
        state = jax.jit(env.step)(state, jnp.zeros(2, dtype=jnp.int32))
        jax.block_until_ready(state)
        assert state.obs.shape == (2, *obs_shape)
        assert state.obs.dtype == jnp.uint8
    finally:
        for env in environments:
            env.close()
