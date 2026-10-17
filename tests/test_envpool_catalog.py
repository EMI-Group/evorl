"""Exercise configured EnvPool tasks through JIT and nested population axes."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from evorl.envs import AutoresetMode, Discrete, create_env

pytestmark = pytest.mark.integration
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs/env/envpool"

# Keep CI bounded while covering every configured family, scalar observations,
# image layouts, observation sizes and continuous/discrete action ranks.
REPRESENTATIVES = {
    "ant-v5.yaml",
    "humanoid-v5.yaml",
    "inverteddoublependulum-v5.yaml",
    "pusher-v5.yaml",
    "reacher-v5.yaml",
    "carracing-v3.yaml",
    "lunarlander-v3.yaml",
    "cartpole-v1.yaml",
    "pendulum-v1.yaml",
    "pong-v5.yaml",
    "catch-v0.yaml",
    "frozenlake-v1.yaml",
    "blackjack-v1.yaml",
    "highway-v0.yaml",
    "racetrack-v1.yaml",
    "reach-v3.yaml",
    "adroithanddoor-v1.yaml",
}


def pytest_generate_tests(metafunc):
    if "envpool_config_path" not in metafunc.fixturenames:
        return
    paths = sorted(CONFIG_ROOT.rglob("*.yaml"))
    if not metafunc.config.getoption("--host-env-catalog"):
        paths = [path for path in paths if path.name in REPRESENTATIVES]
    metafunc.parametrize(
        "envpool_config_path",
        paths,
        ids=[str(p.relative_to(CONFIG_ROOT)) for p in paths],
    )


@pytest.mark.parametrize("host_envs", ["envpool"], indirect=True)
def test_envpool_configured_task(host_envs, envpool_config_path):
    library = pytest.importorskip("envpool")
    config = OmegaConf.load(envpool_config_path)
    if config.env_name not in library.list_all_envs():
        pytest.skip(f"{config.env_name} is unavailable in this EnvPool version")
    _, environments = host_envs
    env = create_env(
        config,
        parallel=1,
        episode_length=3,
        autoreset_mode=AutoresetMode.ENVPOOL,
    )
    native = environments[0]
    observation_shape = native.observation_space.shape
    action_shape = native.action_space.shape
    discrete = isinstance(env.action_space, Discrete)
    actions = jnp.asarray(
        native.action_space.sample(), dtype=jnp.int32 if discrete else jnp.float32
    ).reshape((1, *action_shape))

    def run(key):
        state = env.reset(key)

        def step(state, _):
            state = env.step(state, actions)
            return state, state.reward

        return jax.lax.scan(step, state, None, length=5)

    keys = jax.random.split(jax.random.key(0), (2, 3))
    state, rewards = jax.jit(jax.vmap(jax.vmap(run)))(keys)
    jax.block_until_ready(state)
    assert state.obs.shape == (2, 3, 1, *observation_shape)
    assert state.obs.dtype == jax.dtypes.canonicalize_dtype(
        native.observation_space.dtype
    )
    assert state.reward.shape == state.done.shape == (2, 3, 1)
    assert rewards.shape == (2, 3, 5, 1)
    assert state.reward.dtype == state.done.dtype == jnp.float32
    assert np.isfinite(state.obs).all()
    assert np.isfinite(rewards).all()
    assert environments[-1].num_envs == 6


def test_envpool_config_catalog():
    library = pytest.importorskip("envpool")
    paths = sorted(CONFIG_ROOT.rglob("*.yaml"))
    assert paths
    registered = set(library.list_all_envs())
    seen = set()
    for path in paths:
        config = OmegaConf.load(path)
        assert config.env_type == "envpool", path
        assert config.env_name in registered, path
        assert config.env_name not in seen, path
        assert 0 < config.max_episode_steps < 2147483647, path
        seen.add(config.env_name)


@pytest.mark.parametrize("host_envs", ["envpool"], indirect=True)
@pytest.mark.parametrize("name", ["CartPole-v1", "Hopper-v5", "Pong-v5"])
def test_envpool_nested_matches_flat_pool(host_envs, name):
    library = pytest.importorskip("envpool")
    if name not in library.list_all_envs():
        pytest.skip(f"{name} is unavailable in this EnvPool version")
    env = create_env(
        OmegaConf.create(dict(env_type="envpool", env_name=name)),
        parallel=2,
        episode_length=3,
        seed=123,
    )
    keys = jax.random.split(jax.random.key(0), (2, 3))
    state = jax.jit(jax.vmap(jax.vmap(env.reset)))(keys)
    jax.block_until_ready(state)
    reference = library.make_gymnasium(
        name, num_envs=12, num_threads=2, max_episode_steps=3, seed=123
    )
    expected_obs, _ = reference.reset()
    obs_shape = reference.observation_space.shape
    action_shape = reference.action_space.shape
    np.testing.assert_array_equal(
        state.obs, expected_obs.astype(state.obs.dtype).reshape(2, 3, 2, *obs_shape)
    )
    step = jax.jit(jax.vmap(jax.vmap(env.step)))
    for index in range(5):
        if isinstance(env.action_space, Discrete):
            actions = (np.arange(12, dtype=np.int32) + index) % env.action_space.n
        else:
            fraction = np.random.default_rng(index).uniform(size=(12, *action_shape))
            actions = (
                reference.action_space.low
                + fraction * (reference.action_space.high - reference.action_space.low)
            ).astype(np.float32)
        expected_obs, expected_reward, terminated, truncated, _ = reference.step(
            actions
        )
        state = step(state, jnp.asarray(actions.reshape(2, 3, 2, *action_shape)))
        jax.block_until_ready(state)
        np.testing.assert_array_equal(
            state.obs, expected_obs.astype(state.obs.dtype).reshape(2, 3, 2, *obs_shape)
        )
        np.testing.assert_allclose(
            state.reward, expected_reward.reshape(2, 3, 2), rtol=1e-6
        )
        np.testing.assert_array_equal(
            state.done, np.logical_or(terminated, truncated).reshape(2, 3, 2)
        )
