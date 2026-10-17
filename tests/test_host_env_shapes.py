"""Check array/Dict layouts, terminal observations and host batching."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from evorl.envs import AutoresetMode, create_env
from evorl.envs.space import SpaceContainer
from evorl.types import PyTreeDict


@pytest.fixture
def shape_env_id():
    class ShapeEnv(gym.Env):
        def __init__(
            self, observation="matrix", action="matrix", terminal_on_action=False
        ):
            if observation == "matrix":
                self.observation_space = gym.spaces.Box(-100, 100, (3, 5), np.float32)
            elif observation == "image":
                self.observation_space = gym.spaces.Box(0, 255, (6, 8, 3), np.uint8)
            elif observation == "scalar":
                self.observation_space = gym.spaces.Discrete(7)
            elif observation == "dict":
                self.observation_space = gym.spaces.Dict(
                    value=gym.spaces.Box(-100, 100, (2,), np.float32)
                )
            elif observation in ("nested_dict", "nested_dict_image"):
                nested = dict(
                    matrix=gym.spaces.Box(-100, 100, (3, 5), np.float32),
                    scalar=gym.spaces.Discrete(7),
                )
                if observation == "nested_dict_image":
                    nested["image"] = gym.spaces.Box(0, 255, (6, 8, 3), np.uint8)
                self.observation_space = gym.spaces.Dict(
                    value=gym.spaces.Box(-100, 100, (2,), np.float32),
                    nested=gym.spaces.Dict(nested),
                )
            elif observation == "tuple":
                self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(7),))
            elif observation == "dict_tuple":
                self.observation_space = gym.spaces.Dict(
                    value=gym.spaces.Tuple((gym.spaces.Discrete(7),))
                )
            elif observation == "dict_offset":
                self.observation_space = gym.spaces.Dict(
                    value=gym.spaces.Discrete(7, start=1)
                )
            else:
                raise ValueError(observation)

            if action == "matrix":
                self.action_space = gym.spaces.Box(-1, 1, (2, 3), np.float32)
            elif action == "scalar":
                self.action_space = gym.spaces.Box(-1, 1, (), np.float32)
            elif action == "discrete":
                self.action_space = gym.spaces.Discrete(4)
            elif action == "offset":
                self.action_space = gym.spaces.Discrete(4, start=1)
            elif action == "multidiscrete":
                self.action_space = gym.spaces.MultiDiscrete([2, 3])
            else:
                raise ValueError(action)
            self.value = 0
            self.terminal_on_action = terminal_on_action

        def _observation(self):
            def observe(space):
                if isinstance(space, gym.spaces.Discrete):
                    return np.int64(self.value % space.n + space.start)
                if isinstance(space, gym.spaces.Dict):
                    return {key: observe(child) for key, child in space.spaces.items()}
                if isinstance(space, gym.spaces.Tuple):
                    return tuple(observe(child) for child in space.spaces)
                return np.full(space.shape, self.value, space.dtype)

            return observe(self.observation_space)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.value = 0
            return self._observation(), {}

        def step(self, action):
            self.value += 1
            reward = float(np.asarray(action).sum())
            terminated = self.terminal_on_action and self.value >= 2 and reward > 0
            return self._observation(), reward, terminated, False, {}

    env_id = "EvoRLShapeTest-v0"
    gym.register(env_id, entry_point=ShapeEnv)
    yield env_id
    gym.registry.pop(env_id)


@pytest.mark.parametrize("host_envs", ["gymnasium"], indirect=True)
@pytest.mark.parametrize("observation", ["scalar", "matrix", "image"])
@pytest.mark.parametrize("action", ["scalar", "matrix", "discrete"])
@pytest.mark.parametrize("mode", [AutoresetMode.NORMAL, AutoresetMode.DISABLED])
def test_gymnasium_array_ranks(host_envs, shape_env_id, observation, action, mode):
    _, environments = host_envs
    env = create_env(
        OmegaConf.create(dict(env_type="gymnasium", env_name=shape_env_id)),
        parallel=2,
        episode_length=3,
        autoreset_mode=mode,
        record_ori_obs=mode == AutoresetMode.NORMAL,
        observation=observation,
        action=action,
    )
    native = environments[0]
    obs_shape = native.single_observation_space.shape
    action_shape = native.single_action_space.shape
    dtype = jnp.int32 if action == "discrete" else jnp.float32

    def run(key, actions):
        state = env.reset(key)

        def step(state, _):
            state = env.step(state, actions)
            return state, state.info.truncation

        return jax.lax.scan(step, state, None, length=4)

    keys = jax.random.split(jax.random.key(0), (2, 3))
    # Distinct actions per member/environment catch swaps during flattening.
    actions = jnp.arange(12 * max(1, np.prod(action_shape)), dtype=dtype).reshape(
        (2, 3, 2, *action_shape)
    )
    actions = actions % 4 if action == "discrete" else actions / (1 + actions.size)
    state, truncations = jax.jit(jax.vmap(jax.vmap(run)))(keys, actions)
    jax.block_until_ready(state)
    assert state.obs.shape == (2, 3, 2, *obs_shape)
    assert state.reward.shape == state.done.shape == (2, 3, 2)
    expected_reward = np.asarray(actions).reshape(2, 3, 2, -1).sum(-1)
    np.testing.assert_allclose(state.reward, expected_reward, rtol=1e-6)
    assert state.obs.dtype == jax.dtypes.canonicalize_dtype(
        native.single_observation_space.dtype
    )
    expected = [0, 0, 1, 0] if mode == AutoresetMode.NORMAL else [0, 0, 1, 1]
    np.testing.assert_array_equal(
        truncations,
        np.broadcast_to(np.array(expected)[None, None, :, None], (2, 3, 4, 2)),
    )
    if mode == AutoresetMode.NORMAL:
        assert state.info.ori_obs.shape == state.obs.shape
    assert environments[-1].num_envs == 12


@pytest.mark.parametrize("host_envs", ["gymnasium"], indirect=True)
@pytest.mark.parametrize("observation", ["dict", "nested_dict", "nested_dict_image"])
@pytest.mark.parametrize(
    "mode", [AutoresetMode.NORMAL, AutoresetMode.ENVPOOL, AutoresetMode.DISABLED]
)
def test_gymnasium_dict_terminal_observations(
    host_envs, shape_env_id, observation, mode
):
    env = create_env(
        OmegaConf.create(dict(env_type="gymnasium", env_name=shape_env_id)),
        parallel=2,
        episode_length=3,
        autoreset_mode=mode,
        record_ori_obs=mode == AutoresetMode.NORMAL,
        observation=observation,
        action="discrete",
        terminal_on_action=True,
    )
    assert isinstance(env.obs_space, SpaceContainer)
    initial = jax.jit(env.reset)(jax.random.key(0))
    assert isinstance(initial.obs, PyTreeDict)
    if observation != "nested_dict_image":
        # Space samples must have exactly the observation's PyTree structure.
        sample = jax.jit(env.obs_space.sample)(jax.random.key(1))
        assert jax.tree.structure(sample) == jax.tree.structure(initial.obs)
        assert env.obs_space.contains(sample)
    if mode == AutoresetMode.NORMAL:
        assert jax.tree.structure(initial.obs) == jax.tree.structure(
            initial.info.ori_obs
        )
        for leaf in jax.tree.leaves(initial.info.ori_obs):
            np.testing.assert_array_equal(leaf, np.zeros_like(leaf))

    def run(key, actions):
        def step(state, _):
            state = env.step(state, actions)
            return state, state

        return jax.lax.scan(step, env.reset(key), None, length=5)[1]

    keys = jax.random.split(jax.random.key(0), (2, 3))
    # Worker 0 terminates on step 2; worker 1 reaches the time limit on step 3.
    actions = jnp.broadcast_to(jnp.array([1, 0]), (2, 3, 2))
    states = jax.jit(jax.vmap(jax.vmap(run)))(keys, actions)
    jax.block_until_ready(states)
    if mode == AutoresetMode.NORMAL:
        expected_obs = [[1, 1], [0, 2], [1, 0], [0, 1], [1, 2]]
        expected_terminal = [[1, 1], [2, 2], [1, 3], [2, 1], [1, 2]]
        expected_termination = [[0, 0], [1, 0], [0, 0], [1, 0], [0, 0]]
        expected_truncation = [[0, 0], [0, 0], [0, 1], [0, 0], [0, 0]]
    elif mode == AutoresetMode.ENVPOOL:
        expected_obs = [[1, 1], [2, 2], [0, 3], [1, 0], [2, 1]]
        expected_termination = [[0, 0], [1, 0], [0, 0], [0, 0], [1, 0]]
        expected_truncation = [[0, 0], [0, 0], [0, 1], [0, 0], [0, 0]]
    else:
        expected_obs = [[1, 1], [2, 2], [2, 3], [2, 3], [2, 3]]
        expected_termination = [[0, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
        expected_truncation = [[0, 0], [0, 0], [0, 1], [0, 1], [0, 1]]

    for leaf in jax.tree.leaves(states.obs):
        expected = np.array(expected_obs).reshape((1, 1, 5, 2) + (1,) * (leaf.ndim - 4))
        np.testing.assert_array_equal(leaf, np.broadcast_to(expected, leaf.shape))
    for actual, expected in (
        (states.info.termination, expected_termination),
        (states.info.truncation, expected_truncation),
    ):
        np.testing.assert_array_equal(actual, np.broadcast_to(expected, actual.shape))
    if mode == AutoresetMode.NORMAL:
        for current, terminal in zip(
            jax.tree.leaves(states.obs),
            jax.tree.leaves(states.info.ori_obs),
            strict=True,
        ):
            assert current.dtype == terminal.dtype
            expected = np.array(expected_terminal).reshape(
                (1, 1, 5, 2) + (1,) * (terminal.ndim - 4)
            )
            np.testing.assert_array_equal(
                terminal, np.broadcast_to(expected, terminal.shape)
            )
        assert states.obs.value.dtype == jnp.float32
        if observation == "nested_dict_image":
            assert states.obs.nested.image.dtype == jnp.uint8
            assert states.obs.nested.scalar.dtype == jnp.int32


@pytest.mark.parametrize("host_envs", ["envpool"], indirect=True)
@pytest.mark.parametrize("name", ["AcrobotSwingup-v1", "FetchReach-v4"])
@pytest.mark.parametrize("mode", [AutoresetMode.ENVPOOL, AutoresetMode.DISABLED])
def test_envpool_dict_matches_native(host_envs, name, mode):
    library = pytest.importorskip("envpool")
    if name not in library.list_all_envs():
        pytest.skip(f"{name} is unavailable in this EnvPool version")
    env = create_env(
        OmegaConf.create(dict(env_type="envpool", env_name=name)),
        parallel=2,
        episode_length=3,
        autoreset_mode=mode,
        seed=123,
    )
    assert isinstance(env.obs_space, SpaceContainer)
    sample = jax.jit(env.obs_space.sample)(jax.random.key(1))
    assert env.obs_space.contains(sample)

    def run(key, actions):
        initial = env.reset(key)

        def step(state, _):
            state = env.step(state, actions)
            return state, (
                state.obs,
                state.reward,
                state.done,
                state.info.termination,
                state.info.truncation,
            )

        return initial.obs, jax.lax.scan(step, initial, None, length=5)[1]

    action_shape = env.action_space.shape
    actions = jnp.arange(12 * np.prod(action_shape), dtype=jnp.float32).reshape(
        (2, 3, 2, *action_shape)
    )
    actions = actions / (1 + actions.size)
    keys = jax.random.split(jax.random.key(0), (2, 3))
    initial, states = jax.jit(jax.vmap(jax.vmap(run)))(keys, actions)
    jax.block_until_ready(states)
    assert jax.tree.structure(initial) == jax.tree.structure(sample)

    native = library.make_gymnasium(name, num_envs=12, seed=123, max_episode_steps=3)
    expected_initial, _ = native.reset()
    for actual, expected in zip(
        jax.tree.leaves(initial), jax.tree.leaves(expected_initial), strict=True
    ):
        assert actual.dtype == jax.dtypes.canonicalize_dtype(expected.dtype)
        np.testing.assert_allclose(
            actual.reshape(expected.shape), expected, rtol=1e-6, atol=1e-6
        )
    previous = None
    for i in range(5):
        obs, reward, termination, truncation, _ = native.step(
            np.asarray(actions).reshape((12, *action_shape))
        )
        expected = (obs, reward, termination | truncation, termination, truncation)
        if mode == AutoresetMode.DISABLED and previous is not None:
            done = previous[2]
            expected = jax.tree.map(
                lambda old, new, done=done: np.where(
                    done.reshape((12,) + (1,) * (new.ndim - 1)), old, new
                ),
                previous,
                expected,
            )
        for actual, reference in zip(
            jax.tree.leaves(states), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(actual[:, :, i]).reshape(reference.shape),
                reference,
                rtol=1e-6,
                atol=1e-6,
            )
        previous = jax.tree.map(np.copy, expected)
    assert np.asarray(states[2])[:, :, 2].all()  # Terminal observations at the limit.


@pytest.mark.parametrize("host_envs", ["gymnasium"], indirect=True)
@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(observation="tuple"), "observation space"),
        (dict(observation="dict_tuple"), "observation space"),
        (dict(observation="dict_offset"), "start at zero"),
        (dict(action="multidiscrete"), "action space"),
        (dict(action="offset"), "start at zero"),
        (
            dict(vecenv_kwargs=dict(autoreset_mode=gym.vector.AutoresetMode.DISABLED)),
            "autoreset",
        ),
    ],
)
def test_gymnasium_rejects_unsupported_spaces(host_envs, shape_env_id, kwargs, match):
    with pytest.raises(NotImplementedError, match=match):
        create_env(
            OmegaConf.create(dict(env_type="gymnasium", env_name=shape_env_id)),
            parallel=1,
            **kwargs,
        )


@pytest.mark.parametrize("host_envs", ["envpool"], indirect=True)
@pytest.mark.parametrize(
    "name,kwargs,error,match",
    [
        ("CartPole-v1", dict(max_num_players=2), NotImplementedError, "single-player"),
        ("CartPole-v1", dict(batch_size=1), ValueError, "batch_size"),
    ],
)
def test_envpool_rejects_unsupported_modes(host_envs, name, kwargs, error, match):
    library = pytest.importorskip("envpool")
    if name not in library.list_all_envs():
        pytest.skip(f"{name} is unavailable in this EnvPool version")
    with pytest.raises(error, match=match):
        create_env(
            OmegaConf.create(dict(env_type="envpool", env_name=name)),
            parallel=2,
            **kwargs,
        )


@pytest.mark.parametrize("host_envs", ["envpool"], indirect=True)
def test_envpool_rejects_same_step_autoreset(host_envs, monkeypatch):
    library = pytest.importorskip("envpool")
    original_factory = library.make_gymnasium

    def same_step_factory(*args, **kwargs):
        env = original_factory(*args, **kwargs)
        env.metadata = dict(
            env.metadata, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
        )
        return env

    monkeypatch.setattr(library, "make_gymnasium", same_step_factory)
    with pytest.raises(NotImplementedError, match="next-step"):
        create_env(OmegaConf.create(dict(env_type="envpool", env_name="CartPole-v1")))
