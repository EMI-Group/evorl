"""Scalar/vector host rewards, asynchronous episode boundaries and nested batching."""

from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

pytest.importorskip("mo_gymnasium")

from evorl.envs import AutoresetMode, SpaceContainer, create_env
from evorl.types import PyTreeDict
from evorl.rollout import host_fast_eval_rollout_episode


pytestmark = pytest.mark.integration


@pytest.fixture
def reward_env_id():
    class RewardEnv(gym.Env):
        """Known rewards; positive actions terminate at two steps, zero times out."""

        def __init__(self, objectives=3, structured=False):
            self.objectives = objectives
            self.structured = structured
            self.action_space = gym.spaces.Discrete(3)
            features = gym.spaces.Box(0, 1000, (2,), np.float32)
            self.observation_space = (
                gym.spaces.Dict(
                    features=features,
                    nested=gym.spaces.Dict(count=gym.spaces.Box(0, 1000, (), np.int32)),
                )
                if structured
                else features
            )
            if objectives is not None:
                self.reward_space = gym.spaces.Box(
                    -np.inf, np.inf, (objectives,), np.float32
                )
            self.count = 0

        def _observation(self):
            features = np.array([self.count, self.count + 100], np.float32)
            if self.structured:
                return dict(features=features, nested=dict(count=np.int32(self.count)))
            return features

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.count = 0
            return self._observation(), {}

        def step(self, action):
            self.count += 1
            reward = self.count + 10 * int(action)
            if self.objectives is None:
                reward = float(reward)
            else:
                coordinates = np.array([1, -2, 3], np.float32)[: self.objectives]
                reward = reward * coordinates
            return (
                self._observation(),
                reward,
                self.count == 2 and action > 0,
                False,
                {},
            )

    name = "EvoRLMultiObjectiveRewardTest-v0"
    gym.register(name, entry_point=RewardEnv)
    try:
        yield name
    finally:
        gym.registry.pop(name)


@pytest.mark.parametrize(
    "objectives", [None, 1, 3], ids=["scalar", "vector1", "vector3"]
)
@pytest.mark.parametrize("discount", [1.0, 0.5])
@pytest.mark.parametrize(
    "mode", [AutoresetMode.NORMAL, AutoresetMode.ENVPOOL, AutoresetMode.DISABLED]
)
def test_reward_shapes_returns_and_staggered_nested_autoreset(
    mo_gymnasium_pools, reward_env_id, objectives, discount, mode
):
    env = create_env(
        OmegaConf.create(
            dict(
                env_type="gymnasium" if objectives is None else "mo_gymnasium",
                env_name=reward_env_id,
                max_episode_steps=3,
                objectives=objectives,
            )
        ),
        parallel=2,
        discount=discount,
        record_ori_obs=mode == AutoresetMode.NORMAL,
        autoreset_mode=mode,
    )
    # Ordinary Gymnasium pools are not tracked by the MO fixture.
    try:

        def run(key, actions):
            initial = env.reset(key)

            def step(state, _):
                state = env.step(state, actions)
                return state, state

            return initial, jax.lax.scan(step, initial, (), length=5)[1]

        keys = jax.random.split(jax.random.key(0), (2, 2))
        actions = jnp.array([[[1, 0], [0, 2]], [[2, 1], [1, 2]]], jnp.int32)
        initial, states = jax.jit(jax.vmap(jax.vmap(run)))(keys, actions)
        jax.block_until_ready(states)
        objective_axes = () if objectives is None else (objectives,)
        assert initial.reward.shape == (2, 2, 2, *objective_axes)
        assert states.reward.shape == (2, 2, 5, 2, *objective_axes)
        assert states.reward.dtype == jnp.float32
        assert states.done.shape == states.info.steps.shape == (2, 2, 5, 2)
        np.testing.assert_array_equal(initial.reward, np.zeros_like(initial.reward))

        if mode == AutoresetMode.NORMAL:
            positive_counts, zero_counts = [1, 2, 1, 2, 1], [1, 2, 3, 1, 2]
            positive_term, zero_trunc = [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]
        elif mode == AutoresetMode.ENVPOOL:
            positive_counts, zero_counts = [1, 2, 0, 1, 2], [1, 2, 3, 0, 1]
            positive_term, zero_trunc = [0, 1, 0, 0, 1], [0, 0, 1, 0, 0]
        else:
            positive_counts, zero_counts = [1, 2, 2, 2, 2], [1, 2, 3, 3, 3]
            positive_term, zero_trunc = [0, 1, 1, 1, 1], [0, 0, 1, 1, 1]

        positive = np.asarray(actions)[:, :, None, :] > 0
        counts = np.where(
            positive,
            np.array(positive_counts)[None, None, :, None],
            np.array(zero_counts)[None, None, :, None],
        )
        term = positive * np.array(positive_term)[None, None, :, None]
        trunc = ~positive * np.array(zero_trunc)[None, None, :, None]
        np.testing.assert_array_equal(states.info.steps, counts)
        np.testing.assert_array_equal(states.info.termination, term)
        np.testing.assert_array_equal(states.info.truncation, trunc)
        np.testing.assert_array_equal(states.done, term | trunc)
        obs_counts = (
            np.where(term | trunc, 0, counts)
            if mode == AutoresetMode.NORMAL
            else counts
        )
        np.testing.assert_array_equal(
            states.obs, np.stack((obs_counts, obs_counts + 100), -1)
        )
        if mode == AutoresetMode.NORMAL:
            np.testing.assert_array_equal(
                states.info.ori_obs, np.stack((counts, counts + 100), -1)
            )

        rewards = np.where(
            counts > 0, counts + 10 * np.asarray(actions)[:, :, None, :], 0
        )
        returns = sum(
            np.where(
                counts >= step,
                (step + 10 * np.asarray(actions)[:, :, None, :])
                * discount ** (step - 1),
                0,
            )
            for step in range(1, 4)
        )
        if objectives is not None:
            coordinates = np.array([1, -2, 3])[:objectives]
            rewards = rewards[..., None] * coordinates
            returns = returns[..., None] * coordinates
            assert mo_gymnasium_pools[-1].num_envs == 8
        np.testing.assert_allclose(states.reward, rewards, rtol=1e-6)
        if mode != AutoresetMode.DISABLED:
            assert states.info.episode_return.shape == states.reward.shape
            np.testing.assert_allclose(states.info.episode_return, returns, rtol=1e-6)
        else:
            assert "episode_return" not in states.info
    finally:
        jax.effects_barrier()
        adapter = env.env if mode == AutoresetMode.DISABLED else env
        adapter.env.close()


def test_nested_dict_terminal_observation_and_fast_fallback(
    mo_gymnasium_pools, reward_env_id
):
    with pytest.warns(UserWarning, match="FAST.*Fallback"):
        env = create_env(
            OmegaConf.create(dict(env_type="mo_gymnasium", env_name=reward_env_id)),
            parallel=2,
            episode_length=3,
            objectives=3,
            structured=True,
            autoreset_mode=AutoresetMode.FAST,
            record_ori_obs=True,
        )
    assert isinstance(env.obs_space, SpaceContainer)

    def run(key):
        state = env.reset(key)
        for _ in range(3):
            state = env.step(state, jnp.array([1, 0]))
        return state

    states = jax.jit(jax.vmap(jax.vmap(run)))(
        jax.random.split(jax.random.key(2), (2, 2))
    )
    jax.block_until_ready(states)
    assert isinstance(states.obs, PyTreeDict)
    np.testing.assert_array_equal(
        states.obs.nested.count, np.broadcast_to([1, 0], (2, 2, 2))
    )
    np.testing.assert_array_equal(
        states.info.ori_obs.nested.count, np.broadcast_to([1, 3], (2, 2, 2))
    )
    np.testing.assert_array_equal(states.obs.features[..., 0], states.obs.nested.count)
    np.testing.assert_array_equal(
        states.info.ori_obs.features[..., 0], states.info.ori_obs.nested.count
    )
    assert (
        states.obs.nested.count.dtype
        == states.info.ori_obs.nested.count.dtype
        == jnp.int32
    )
    assert states.reward.shape == (2, 2, 2, 3)


@pytest.mark.parametrize(
    "mode", [AutoresetMode.NORMAL, AutoresetMode.ENVPOOL, AutoresetMode.DISABLED]
)
def test_real_deep_sea_treasure_matches_native_transitions(mo_gymnasium_pools, mode):
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs/env/mo_gymnasium/deep-sea-treasure-v0.yaml")
    env = create_env(
        config,
        parallel=2,
        episode_length=3,
        autoreset_mode=mode,
        record_ori_obs=mode == AutoresetMode.NORMAL,
    )

    def run(key):
        def step(state, _):
            state = env.step(state, jnp.array([1, 3]))
            return state, state

        return jax.lax.scan(step, env.reset(key), (), length=4)[1]

    states = jax.jit(jax.vmap(jax.vmap(run)))(
        jax.random.split(jax.random.key(3), (2, 2))
    )
    jax.block_until_ready(states)
    np.testing.assert_allclose(
        states.reward[:, :, 0], np.broadcast_to([[0.7, -1], [0, -1]], (2, 2, 2, 2))
    )
    np.testing.assert_array_equal(
        states.info.termination[:, :, 0], np.broadcast_to([1, 0], (2, 2, 2))
    )
    np.testing.assert_array_equal(states.info.truncation[:, :, 2, 1], 1)
    if mode == AutoresetMode.NORMAL:
        np.testing.assert_array_equal(states.obs[:, :, 0, 0], 0)
        np.testing.assert_array_equal(
            states.info.ori_obs[:, :, 0, 0], np.broadcast_to([1, 0], (2, 2, 2))
        )
        np.testing.assert_array_equal(
            states.info.ori_obs[:, :, 2, 1], np.broadcast_to([0, 3], (2, 2, 2))
        )
    else:
        np.testing.assert_array_equal(
            states.obs[:, :, 0, 0], np.broadcast_to([1, 0], (2, 2, 2))
        )


def test_vector_host_evaluation_stops_at_each_native_episode(
    mo_gymnasium_pools, reward_env_id
):
    env = create_env(
        OmegaConf.create(dict(env_type="mo_gymnasium", env_name=reward_env_id)),
        parallel=2,
        episode_length=3,
        autoreset_mode=AutoresetMode.DISABLED,
        objectives=3,
    )

    def actions(agent_state, batch, key):
        return jnp.array([1, 0]), PyTreeDict()

    def evaluate(key):
        return host_fast_eval_rollout_episode(
            env.step, actions, env.reset(key), None, key, 5
        )

    metrics, states = jax.jit(jax.vmap(jax.vmap(evaluate)))(
        jax.random.split(jax.random.key(4), (2, 2))
    )
    jax.block_until_ready(states)
    np.testing.assert_array_equal(
        metrics.episode_lengths, np.broadcast_to([2, 3], (2, 2, 2))
    )
    np.testing.assert_allclose(
        metrics.episode_returns,
        np.broadcast_to(np.array([23, 6])[:, None] * [1, -2, 3], (2, 2, 2, 3)),
    )
