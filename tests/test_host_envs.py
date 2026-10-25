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
from evorl.rollout import host_fast_eval_rollout_episode
from evorl.types import PyTreeDict


@pytest.mark.parametrize("host_envs", ["envpool"], indirect=True)
@pytest.mark.parametrize("nested", [False, True])
def test_envpool_original_observations_at_termination(host_envs, nested):
    """The factory consumes record_ori_obs and retains next-step terminal obs."""
    env = create_env(
        OmegaConf.create(dict(env_type="envpool", env_name="Pendulum-v1")),
        parallel=2,
        episode_length=2,
        autoreset_mode=AutoresetMode.ENVPOOL,
        record_ori_obs=True,
    )

    def run(key):
        state = env.reset(key)

        def step(state, _):
            state = env.step(state, jnp.zeros((2, 1)))
            return state, (
                state.obs,
                state.info.ori_obs,
                state.info.truncation,
                state.info.autoreset,
            )

        return jax.lax.scan(step, state, None, length=3)[1]

    if nested:
        result = jax.jit(jax.vmap(jax.vmap(run)))(
            jax.random.split(jax.random.key(0), (2, 3))
        )
        expected_shape = (2, 3, 3, 2, 3)
    else:
        result = jax.jit(run)(jax.random.key(0))
        expected_shape = (3, 2, 3)
    obs, original, truncation, autoreset = jax.block_until_ready(result)
    assert original.shape == obs.shape == expected_shape
    np.testing.assert_array_equal(original, obs)
    np.testing.assert_array_equal(
        truncation, np.broadcast_to(np.array([0, 1, 0])[:, None], truncation.shape)
    )
    np.testing.assert_array_equal(
        autoreset, np.broadcast_to(np.array([0, 0, 1])[:, None], autoreset.shape)
    )


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
        if backend == "envpool":
            adapter = env.env if autoreset_mode == AutoresetMode.DISABLED else env
            assert adapter.autoreset_next_step
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


@pytest.mark.parametrize("env_name,obs_size", [("CartPole-v1", 4), ("Pendulum-v1", 3)])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)], ids=["batch", "nested_vmap"])
@pytest.mark.parametrize(
    "autoreset_mode", [AutoresetMode.ENVPOOL, AutoresetMode.DISABLED]
)
def test_host_env_reset_and_scan(
    host_envs, env_name, obs_size, batch_shape, autoreset_mode
):
    backend, environments = host_envs
    env = create_env(
        OmegaConf.create(dict(env_type=backend, env_name=env_name)),
        parallel=1,
        episode_length=3,
        autoreset_mode=autoreset_mode,
    )

    def trajectory(key, actions):
        state = env.reset(key)

        def body(state, action):
            state = env.step(state, action)
            return state, (state.obs, state.info.truncation)

        # Constant actions never read reset's observation. Reset must still
        # execute, and must precede the first step of this compiled rollout.
        return jax.lax.scan(body, state, actions)[1]

    key = jax.random.key(0)
    keys = jax.random.split(key, batch_shape) if batch_shape else key
    for _ in batch_shape:
        trajectory = jax.vmap(trajectory)
    dtype = jnp.int32 if env_name == "CartPole-v1" else jnp.float32
    actions = jnp.zeros((*batch_shape, 6, 1, *env.action_space.shape), dtype)
    obs, truncated = jax.jit(trajectory)(keys, actions)
    jax.block_until_ready(obs)
    assert obs.shape == (*batch_shape, 6, 1, obs_size)
    expected = (
        [0, 0, 1, 1, 1, 1]
        if autoreset_mode == AutoresetMode.DISABLED
        else [0, 0, 1, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        truncated, np.broadcast_to(np.array(expected)[:, None], (*batch_shape, 6, 1))
    )
    assert len(environments) == 2
    assert environments[-1].num_envs == (np.prod(batch_shape) if batch_shape else 1)


def test_host_callbacks_execute_when_results_are_unused(host_envs, monkeypatch):
    backend, environments = host_envs
    env = create_env(
        OmegaConf.create(dict(env_type=backend, env_name="CartPole-v1")),
        parallel=1,
        episode_length=3,
        autoreset_mode=AutoresetMode.ENVPOOL,
    )

    @jax.jit
    def discarded_reset(key):
        env.reset(key)
        return jnp.int32(0)

    discarded_reset(jax.random.key(0)).block_until_ready()
    jax.effects_barrier()
    assert len(environments) == 2
    state = jax.jit(env.reset)(jax.random.key(1))
    jax.block_until_ready(state)
    native = environments[-1]
    original_step = native.step
    calls = []

    def counted_step(actions):
        calls.append(1)
        return original_step(actions)

    monkeypatch.setattr(native, "step", counted_step)

    @jax.jit
    def discarded_step(state):
        env.step(state, jnp.zeros(1, jnp.int32))
        return jnp.int32(0)

    discarded_step(state).block_until_ready()
    jax.effects_barrier()
    assert calls == [1]


def test_host_episode_resets_stay_ordered(host_envs, monkeypatch):
    """A second reset must not replace the pool while episode one is running."""
    backend, environments = host_envs
    library = importlib.import_module(backend)
    factory_name = "make_vec" if backend == "gymnasium" else "make_gymnasium"
    original_factory = getattr(library, factory_name)
    events = []

    def tracked_factory(*args, **kwargs):
        native = original_factory(*args, **kwargs)
        pool_id = len(environments) - 1
        reset, step = native.reset, native.step

        def tracked_reset(*args, **kwargs):
            events.append(("reset", pool_id))
            return reset(*args, **kwargs)

        def tracked_step(*args, **kwargs):
            events.append(("step", pool_id))
            return step(*args, **kwargs)

        monkeypatch.setattr(native, "reset", tracked_reset)
        monkeypatch.setattr(native, "step", tracked_step)
        return native

    monkeypatch.setattr(library, factory_name, tracked_factory)
    env = create_env(
        OmegaConf.create(dict(env_type=backend, env_name="CartPole-v1")),
        parallel=1,
        episode_length=3,
        autoreset_mode=AutoresetMode.DISABLED,
    )

    def action_fn(agent_state, batch, key):
        return jnp.zeros(1, jnp.int32), PyTreeDict()

    def episode(key):
        return host_fast_eval_rollout_episode(
            env.step, action_fn, env.reset(key), None, key, 3
        )

    @jax.jit
    def two_episodes(first_key, second_key):
        first = episode(first_key)
        second = episode(second_key)
        return first, second

    # The initial pool is used only to construct callback output specifications.
    events.clear()
    first, second = two_episodes(jax.random.key(1), jax.random.key(2))
    jax.block_until_ready((first, second))
    jax.effects_barrier()
    assert events == [
        ("reset", 1),
        ("step", 1),
        ("step", 1),
        ("step", 1),
        ("reset", 2),
        ("step", 2),
        ("step", 2),
        ("step", 2),
    ]
    for metrics, state in (first, second):
        np.testing.assert_array_equal(metrics.episode_lengths, [3])
        np.testing.assert_array_equal(state.done, [1])


@pytest.mark.parametrize("env_name", ["CartPole-v1", "Pendulum-v1"])
def test_host_same_step_nested_vmap(host_envs, env_name):
    backend, _ = host_envs
    env = create_env(
        OmegaConf.create(dict(env_type=backend, env_name=env_name)),
        parallel=1,
        episode_length=2,
        autoreset_mode=AutoresetMode.NORMAL,
        record_ori_obs=True,
    )

    def run(key):
        state = env.reset(key)
        action = jnp.zeros(
            (1, *env.action_space.shape),
            jnp.int32 if env_name == "CartPole-v1" else jnp.float32,
        )
        for _ in range(2):
            state = env.step(state, action)
        return state

    keys = jax.random.split(jax.random.key(0), (2, 2))
    state = jax.jit(jax.vmap(jax.vmap(run)))(keys)
    jax.block_until_ready(state)
    np.testing.assert_array_equal(state.info.truncation, np.ones((2, 2, 1)))
    assert state.info.ori_obs.shape == state.obs.shape
    assert np.any(np.asarray(state.info.ori_obs) != np.asarray(state.obs))


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
