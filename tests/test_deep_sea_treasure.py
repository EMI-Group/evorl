"""Classic treasure paths, collision behavior and EvoRL episode wrappers."""

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.envs import AutoresetMode, create_env
from evorl.envs.toy import DeepSeaTreasure


@pytest.mark.parametrize(
    "row,column,value",
    [
        (1, 0, 0.7),
        (2, 1, 8.2),
        (3, 2, 11.5),
        (4, 3, 14.0),
        (4, 4, 15.1),
        (4, 5, 16.1),
        (7, 6, 19.6),
        (7, 7, 20.3),
        (9, 8, 22.4),
        (10, 9, 23.7),
    ],
)
def test_classic_treasure_paths_preserve_value_and_time(row, column, value, jit):
    env = DeepSeaTreasure()

    def run(key):
        state = env.reset(key)

        def step(state, action):
            state = env.step(state, action)
            return state, state.reward

        return jax.lax.scan(step, state, jnp.array([3] * column + [1] * row))

    state, rewards = (jax.jit(run) if jit else run)(jax.random.key(0))
    np.testing.assert_allclose(rewards.sum(0), [value, -(row + column)], atol=1e-5)
    np.testing.assert_array_equal(state.obs, [row, column])
    assert state.done == 1


def test_grid_boundaries_and_rocks_block_movement():
    env = DeepSeaTreasure()
    state = env.reset(jax.random.key(0))
    np.testing.assert_array_equal(env.step(state, 0).obs, [0, 0])
    np.testing.assert_array_equal(env.step(state, 2).obs, [0, 0])
    # (2, 1) is a treasure; (2, 0) is a rock. Set a valid position directly
    # to test collision independently of episode termination.
    state = state.replace(env_state=jnp.array([3, 2]), obs=jnp.array([3, 2]))
    blocked = env.step(state, 2)
    np.testing.assert_array_equal(blocked.obs, [3, 2])


@pytest.mark.parametrize("mode", list(AutoresetMode))
def test_treasure_nested_autoreset_preserves_terminal_vector_reward(mode):
    config = OmegaConf.create(
        dict(env_type="toy", env_name="deep-sea-treasure-v0", max_episode_steps=2)
    )
    env = create_env(config, parallel=2, autoreset_mode=mode, record_ori_obs=True)

    def run(key):
        state = env.reset(key)

        def step(state, _):
            state = env.step(state, jnp.ones(2, jnp.int32))
            return state, state.reward

        return jax.lax.scan(step, state, (), length=3)

    keys = jax.random.split(jax.random.key(1), (2, 2))
    states, rewards = jax.jit(jax.vmap(jax.vmap(run)))(keys)
    assert rewards.shape == (2, 2, 3, 2, 2)
    np.testing.assert_allclose(rewards[:, :, 0, :, 0], 0.7)
    assert np.all(np.isfinite(rewards))
    assert states.obs.shape == (2, 2, 2, 2)
