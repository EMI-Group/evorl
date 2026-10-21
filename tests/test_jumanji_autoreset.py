"""Check EvoRL autoreset modes against independent episode references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jumanji = pytest.importorskip("jumanji")

from evorl.envs import AutoresetMode, MultiAgentEnv, create_env
from evorl.envs.jumanji import create_wrapped_jumanji_env
from evorl.types import PyTreeDict
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
    assert "discount" not in state.info


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
                assert "discount" not in actual.info
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


@pytest.mark.parametrize(
    "env_name", ["Maze-v0", "LevelBasedForaging-v0", "Connector-v3", "Game2048-v1"]
)
@pytest.mark.parametrize("mode", list(AutoresetMode))
@pytest.mark.integration
def test_shared_time_limit_preserves_native_end_type(env_name, mode):
    from jumanji.environments.routing.maze.generator import ToyGenerator

    kwargs = {"generator": ToyGenerator()} if env_name == "Maze-v0" else {}
    env = create_env(
        PyTreeDict(env_type="jumanji", env_name=env_name, max_episode_steps=4),
        parallel=2,
        autoreset_mode=mode,
        record_ori_obs=True,
        **kwargs,
    )
    reset = jax.jit(jax.vmap(jax.vmap(env.reset)))
    step = jax.jit(jax.vmap(jax.vmap(env.step)))
    state = reset(jax.random.split(jax.random.key(0), (2, 2)))
    multi_agent = isinstance(env.unwrapped, MultiAgentEnv)
    actions = (
        {
            agent: jnp.zeros((2, 2, 2, *space.shape), dtype=jnp.int32)
            for agent, space in env.action_space.items()
        }
        if multi_agent
        else jnp.zeros((2, 2, 2), dtype=jnp.int32)
    )
    for _ in range(3):
        state = step(state, actions)
        episode_done = state.done["__all__"] if multi_agent else state.done
        np.testing.assert_array_equal(episode_done, 0)
    state = step(state, actions)
    episode_done = state.done["__all__"] if multi_agent else state.done
    np.testing.assert_array_equal(episode_done, jnp.ones((2, 2, 2)))
    terminated = int(env_name == "Maze-v0")
    for leaf in jax.tree.leaves(state.info.termination):
        np.testing.assert_array_equal(leaf, terminated)
    for leaf in jax.tree.leaves(state.info.truncation):
        np.testing.assert_array_equal(leaf, 1 - terminated)
    np.testing.assert_array_equal(state.info.steps, 4)
    assert "truncation" not in state._internal
    assert "termination" not in state._internal
    assert "discount" not in state.info
    if mode != AutoresetMode.ENVPOOL and env_name != "Game2048-v1":
        observations = (
            state.info.ori_obs.values() if multi_agent else [state.info.ori_obs]
        )
        for obs in observations:
            np.testing.assert_array_equal(obs.step_count, 4)


@pytest.mark.integration
@pytest.mark.parametrize("episode_length", [2, 4])
def test_maze_success_remains_termination(episode_length):
    from dataclasses import replace
    from jumanji.environments.routing.maze.generator import ToyGenerator
    from jumanji.environments.routing.maze.types import Position

    class NearTargetGenerator(ToyGenerator):
        def __call__(self, key):
            return replace(super().__call__(key), agent_position=Position(0, 2))

    env = create_wrapped_jumanji_env(
        "Maze-v0",
        generator=NearTargetGenerator(),
        episode_length=episode_length,
        parallel=1,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    state = jax.jit(env.reset)(jax.random.key(0))
    step = jax.jit(env.step)
    state = step(state, jnp.ones(1, jnp.int32))
    np.testing.assert_array_equal(state.done, 0)
    state = step(state, jnp.ones(1, jnp.int32))
    np.testing.assert_array_equal(state.done, 1)
    np.testing.assert_array_equal(state.info.termination, 1)
    np.testing.assert_array_equal(state.info.truncation, 0)


@pytest.mark.integration
@pytest.mark.parametrize("episode_length", [1, 2])
def test_foraging_success_takes_priority_over_timeout(episode_length):
    from jumanji.environments.routing.lbf.generator import RandomGenerator

    class NearFoodGenerator(RandomGenerator):
        def __call__(self, key):
            state = super().__call__(key)
            return state.replace(
                agents=state.agents.replace(
                    position=jnp.array([[1, 1], [1, 3]]),
                    level=jnp.ones_like(state.agents.level),
                ),
                food_items=state.food_items.replace(
                    position=jnp.array([[1, 2]]),
                    level=jnp.ones_like(state.food_items.level),
                ),
            )

    env = create_wrapped_jumanji_env(
        "LevelBasedForaging-v0",
        generator=NearFoodGenerator(grid_size=8, num_agents=2, num_food=1, fov=8),
        episode_length=episode_length,
        parallel=1,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    state = jax.jit(env.reset)(jax.random.key(0))
    actions = {agent: jnp.full(1, 5, jnp.int32) for agent in env.action_space}
    state = jax.jit(env.step)(state, actions)
    for leaf in jax.tree.leaves((state.done, state.info.termination)):
        np.testing.assert_array_equal(leaf, 1)
    for leaf in jax.tree.leaves(state.info.truncation):
        np.testing.assert_array_equal(leaf, 0)


@pytest.mark.integration
def test_connector_keeps_individual_termination_at_timeout(monkeypatch):
    env = create_wrapped_jumanji_env(
        "Connector-v3",
        episode_length=1,
        parallel=2,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    native = env.unwrapped.env
    native_step = native.step

    def step(state, action):
        state, timestep = native_step(state, action)
        return state, timestep.replace(
            discount=jnp.ones(native.num_agents).at[0].set(0)
        )

    monkeypatch.setattr(native, "step", step)
    state = jax.jit(env.reset)(jax.random.key(0))
    actions = {agent: jnp.zeros(2, jnp.int32) for agent in env.action_space}
    state = jax.jit(env.step)(state, actions)
    np.testing.assert_array_equal(state.info.termination.agent_0, 1)
    np.testing.assert_array_equal(state.info.truncation.agent_0, 0)
    np.testing.assert_array_equal(state.info.termination.agent_1, 0)
    np.testing.assert_array_equal(state.info.truncation.agent_1, 1)
    np.testing.assert_array_equal(state.info.termination["__all__"], 0)
    np.testing.assert_array_equal(state.info.truncation["__all__"], 1)


def test_mmst_keeps_caller_provided_generator():
    native = jumanji.make("MMST-v0", time_limit=8)
    generator = native._generator
    env = create_wrapped_jumanji_env(
        "MMST-v0",
        generator=generator,
        episode_length=6,
        parallel=1,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    state = jax.eval_shape(env.reset, jax.random.key(0))
    assert state.env_state.connected_nodes.shape[-1] == 8
    assert generator._max_step == 8
    assert env.unwrapped.env._generator is generator
    assert env.unwrapped.env.observation_spec.step_count.maximum == 6


def test_search_and_rescue_uses_native_reward_function_and_shared_horizon():
    from jumanji.environments.swarms.search_and_rescue.reward import IndividualRewardFn

    reward_fn = IndividualRewardFn()
    env = create_wrapped_jumanji_env(
        "SearchAndRescue-v0",
        reward_fn=reward_fn,
        episode_length=6,
        parallel=1,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    native = env.unwrapped.env
    assert native._reward_fn is reward_fn
    assert native.time_limit == 6
    found = jnp.array([[True, False], [False, True]])
    for step in range(6):
        reward = native._reward_fn(found, step, native.time_limit)
        np.testing.assert_allclose(reward, reward_fn(found, step, 6))
        assert jnp.all(reward >= 0)


@pytest.mark.parametrize(
    "env_name",
    [
        "Maze-v0",
        "Cleaner-v0",
        "Connector-v3",
        "LevelBasedForaging-v0",
        "MMST-v0",
        "RobotWarehouse-v0",
        "SearchAndRescue-v0",
        "RubiksCube-v0",
        "RubiksCube-partly-scrambled-v0",
        "SlidingTilePuzzle-v0",
        "Tetris-v0",
        "PacMan-v1",
        "Snake-v1",
        "Sokoban-v0",
    ],
)
def test_constructor_receives_shared_time_limit(env_name):
    from jumanji.environments.routing.sokoban.generator import ToyGenerator

    kwargs = {"generator": ToyGenerator()} if env_name == "Sokoban-v0" else {}
    env = create_wrapped_jumanji_env(env_name, episode_length=6, parallel=1, **kwargs)
    native = env.unwrapped.env
    assert native.time_limit == env.episode_length == 6
    spec = native.observation_spec
    if hasattr(spec, "step_count"):
        if hasattr(spec.step_count, "num_values"):
            assert spec.step_count.num_values == 6
        elif hasattr(spec.step_count, "maximum"):
            assert spec.step_count.maximum == 6
    elif hasattr(spec, "step"):
        assert spec.step.maximum == 6
    if env_name == "MMST-v0":
        assert (
            jax.eval_shape(native.reset, jax.random.key(0))[0].connected_nodes.shape[-1]
            == 6
        )


def test_conflicting_time_limits_are_rejected():
    with pytest.raises(ValueError, match="time_limit must match"):
        create_wrapped_jumanji_env("Maze-v0", episode_length=6, time_limit=2)


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
