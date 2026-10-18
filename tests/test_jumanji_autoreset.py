"""Check EvoRL autoreset modes against independent episode references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jumanji = pytest.importorskip("jumanji")

from evorl.envs import AutoresetMode
from evorl.envs.jumanji import create_wrapped_jumanji_env
from tests.helpers.jumanji import CounterEnv


@pytest.mark.parametrize("mode", list(AutoresetMode))
def test_episode_boundaries_and_terminal_observation(monkeypatch, mode):
    monkeypatch.setattr(jumanji, "make", lambda *args, **kwargs: CounterEnv())
    env = create_wrapped_jumanji_env(
        "Counter",
        episode_length=2,
        parallel=2,
        autoreset_mode=mode,
        record_ori_obs=True,
    )
    state = jax.jit(env.reset)(jax.random.key(0))
    step = jax.jit(env.step)
    action = jnp.zeros(2, dtype=jnp.int32)
    state = step(state, action)
    np.testing.assert_array_equal(state.done, [0, 0])
    terminal = step(state, action)
    np.testing.assert_array_equal(terminal.done, [1, 1])
    np.testing.assert_array_equal(terminal.info.steps, [2, 2])
    np.testing.assert_array_equal(terminal.info.termination, [0, 0])
    np.testing.assert_array_equal(terminal.info.truncation, [1, 1])
    if mode != AutoresetMode.ENVPOOL:
        np.testing.assert_array_equal(terminal.info.ori_obs.value, [[2], [2]])
    if mode in (AutoresetMode.NORMAL, AutoresetMode.FAST):
        np.testing.assert_array_equal(terminal.obs.value, [[0], [0]])
        np.testing.assert_array_equal(terminal.info.episode_return, [2, 2])
    else:
        np.testing.assert_array_equal(terminal.obs.value, [[2], [2]])
    next_state = step(terminal, action)
    if mode == AutoresetMode.DISABLED:
        assert jax.tree.all(jax.tree.map(jnp.array_equal, terminal, next_state))
    elif mode == AutoresetMode.ENVPOOL:
        np.testing.assert_array_equal(next_state.info.autoreset, [1, 1])
        np.testing.assert_array_equal(next_state.obs.value, [[0], [0]])
        np.testing.assert_array_equal(next_state.reward, [0, 0])
    else:
        np.testing.assert_array_equal(next_state.info.steps, [1, 1])
        np.testing.assert_array_equal(next_state.done, [0, 0])


@pytest.mark.parametrize("episode_length", [3, 10])
def test_native_termination_is_preserved(monkeypatch, episode_length):
    monkeypatch.setattr(jumanji, "make", lambda *args, **kwargs: CounterEnv())
    env = create_wrapped_jumanji_env(
        "Counter",
        episode_length=episode_length,
        parallel=1,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    state = jax.jit(env.reset)(jax.random.key(0))
    step = jax.jit(env.step)
    for _ in range(3):
        state = step(state, jnp.zeros(1, dtype=jnp.int32))
    np.testing.assert_array_equal(state.done, [1])
    np.testing.assert_array_equal(state.info.termination, [1])
    np.testing.assert_array_equal(state.info.truncation, [0])
    np.testing.assert_array_equal(state.info.discount, [0])


@pytest.mark.parametrize("mode", list(AutoresetMode))
@pytest.mark.parametrize(
    "prebatched", [False, True], ids=["split_keys", "batched_keys"]
)
def test_staggered_nested_episodes_match_reference(monkeypatch, mode, prebatched):
    from tests.helpers.jumanji import StaggeredEnv

    native = StaggeredEnv()
    monkeypatch.setattr(jumanji, "make", lambda *args, **kwargs: native)
    env = create_wrapped_jumanji_env(
        "Staggered",
        episode_length=3,
        parallel=3,
        autoreset_mode=mode,
        discount=0.9,
        record_ori_obs=True,
    )
    native_reset = jax.jit(native.reset)
    native_step = jax.jit(native.step)
    keys = jax.random.split(jax.random.key(7), (2, 2))

    def collect(key):
        reset_key = jax.random.split(key, 3) if prebatched else key
        state = env.reset(reset_key)

        def step(state, _):
            state = env.step(state, jnp.zeros(3, dtype=jnp.int32))
            return state, state

        return jax.lax.scan(step, state, None, length=8)[1]

    history = jax.jit(jax.vmap(jax.vmap(collect)))(keys)
    # At least one batch step contains both finished and unfinished environments.
    assert jnp.any(jnp.any(history.done, axis=-1) & ~jnp.all(history.done, axis=-1))
    for outer in np.ndindex(2, 2):
        for slot, key in enumerate(jax.random.split(keys[outer], 3)):
            if mode in (AutoresetMode.NORMAL, AutoresetMode.ENVPOOL):
                reset_key, key = jax.random.split(key)
            else:
                reset_key = key
            state, transition = native_reset(key)
            first_state, first_transition = state, transition
            done, count, episode_return = False, 0, 0.0
            last_actual = None
            for index in range(8):
                position = (*outer, index, slot)
                actual = jax.tree.map(lambda x, pos=position: x[pos], history)
                if done and mode == AutoresetMode.DISABLED:
                    assert jax.tree.all(
                        jax.tree.map(jnp.array_equal, actual, last_actual)
                    )
                    continue
                if done:
                    count, episode_return = 0, 0.0
                autoreset = done and mode == AutoresetMode.ENVPOOL
                if autoreset:
                    reset_key, key = jax.random.split(reset_key)
                    state, transition = native_reset(key)
                    done, termination, truncation = False, False, False
                    terminal_obs = transition.observation
                else:
                    state, transition = native_step(state, jnp.array(0))
                    count += 1
                    termination = bool(transition.last())
                    truncation = count >= 3 and not termination
                    done = termination or truncation
                    episode_return += 0.9 ** (count - 1) * float(transition.reward)
                    terminal_obs = transition.observation
                    if done and mode == AutoresetMode.NORMAL:
                        reset_key, key = jax.random.split(reset_key)
                        state, reset_transition = native_reset(key)
                    elif done and mode == AutoresetMode.FAST:
                        state, reset_transition = first_state, first_transition
                obs = (
                    reset_transition.observation
                    if (done and mode in (AutoresetMode.NORMAL, AutoresetMode.FAST))
                    else transition.observation
                )
                np.testing.assert_array_equal(actual.obs.value, obs.value)
                np.testing.assert_array_equal(actual.obs.reset_marker, obs.reset_marker)
                np.testing.assert_array_equal(actual.obs.action_mask, obs.action_mask)
                assert jax.tree.all(
                    jax.tree.map(jnp.array_equal, actual.env_state, state)
                )
                np.testing.assert_allclose(actual.reward, transition.reward)
                assert actual.done == done
                assert actual.info.steps == count
                assert actual.info.termination == termination
                assert actual.info.truncation == truncation
                assert actual.info.counter == transition.extras["counter"]
                np.testing.assert_allclose(actual.info.discount, transition.discount)
                if mode == AutoresetMode.ENVPOOL:
                    assert actual.info.autoreset == autoreset
                else:
                    np.testing.assert_array_equal(
                        actual.info.ori_obs.value, terminal_obs.value
                    )
                    np.testing.assert_array_equal(
                        actual.info.ori_obs.action_mask, terminal_obs.action_mask
                    )
                if mode != AutoresetMode.DISABLED:
                    np.testing.assert_allclose(
                        actual.info.episode_return, episode_return
                    )
                last_actual = actual


@pytest.mark.parametrize("env_name", ["LevelBasedForaging-v0", "Connector-v3"])
@pytest.mark.parametrize("mode", list(AutoresetMode))
@pytest.mark.integration
def test_native_truncation_stays_truncation(env_name, mode):
    env = create_wrapped_jumanji_env(
        env_name,
        time_limit=2,
        episode_length=10,
        parallel=2,
        autoreset_mode=mode,
        record_ori_obs=True,
    )
    reset = jax.jit(jax.vmap(jax.vmap(env.reset)))
    step = jax.jit(jax.vmap(jax.vmap(env.step)))
    state = reset(jax.random.split(jax.random.key(0), (2, 2)))
    actions = {
        agent: jnp.zeros((2, 2, 2, *space.shape), dtype=jnp.int32)
        for agent, space in env.action_space.items()
    }
    state = step(state, actions)
    state = step(state, actions)
    np.testing.assert_array_equal(state.done["__all__"], jnp.ones((2, 2, 2)))
    for leaf in jax.tree.leaves(state.info.termination):
        np.testing.assert_array_equal(leaf, jnp.zeros((2, 2, 2)))
    for leaf in jax.tree.leaves((state.info.truncation, state.info.discount)):
        np.testing.assert_array_equal(leaf, jnp.ones((2, 2, 2)))
    if mode != AutoresetMode.ENVPOOL:
        for obs in state.info.ori_obs.values():
            np.testing.assert_array_equal(obs.step_count, 2)


@pytest.mark.parametrize("mode", list(AutoresetMode))
def test_only_evorl_wrappers_handle_autoreset(mode):
    from jumanji.wrappers import (
        AutoResetWrapper,
        VmapAutoResetWrapper as NativeVmapAutoReset,
    )
    from evorl.envs.wrappers.training_wrapper import (
        FastVmapAutoResetWrapper,
        OneEpisodeWrapper,
        VmapAutoResetWrapper,
        VmapEnvPoolAutoResetWrapper,
    )
    from evorl.envs.wrappers.wrapper import get_wrapper

    env = create_wrapped_jumanji_env("Game2048-v1", parallel=2, autoreset_mode=mode)
    expected = {
        AutoresetMode.NORMAL: VmapAutoResetWrapper,
        AutoresetMode.FAST: FastVmapAutoResetWrapper,
        AutoresetMode.ENVPOOL: VmapEnvPoolAutoResetWrapper,
        AutoresetMode.DISABLED: OneEpisodeWrapper,
    }
    assert get_wrapper(env, expected[mode]) is not None
    while hasattr(env, "env"):
        assert not isinstance(env, (AutoResetWrapper, NativeVmapAutoReset))
        env = env.env
