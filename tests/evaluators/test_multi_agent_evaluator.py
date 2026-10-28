"""Per-agent episode returns through the public evaluator API."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.envs.wrappers import OneEpisodeWrapper, VmapWrapper
from evorl.evaluators import MultiAgentEvaluator
from evorl.types import PyTreeDict
from tests.helpers.multi_agent_envs import ToyMultiAgentEnv


class _MixedRewardEnv(ToyMultiAgentEnv):
    """The first agent has vector rewards; the second keeps scalar rewards."""

    def __init__(self, num_objectives):
        super().__init__(continuous=True)
        self.num_objectives = num_objectives

    def reset(self, key):
        state = super().reset(key)
        return state.replace(
            reward=state.reward.replace(first=jnp.zeros(self.num_objectives))
        )

    def step(self, state, actions):
        state = super().step(state, actions)
        return state.replace(
            reward=state.reward.replace(
                first=jnp.arange(1, self.num_objectives + 1, dtype=jnp.float32)
            )
        )


@pytest.mark.parametrize("num_objectives", [1, 3])
@pytest.mark.parametrize("discount", [0.5, 1.0])
@pytest.mark.parametrize("nested", [False, True])
def test_mixed_scalar_and_vector_returns_keep_episode_and_objective_axes(
    num_objectives, discount, nested, jit
):
    env = VmapWrapper(
        OneEpisodeWrapper(_MixedRewardEnv(num_objectives), 3), 2, vmap_step=True
    )

    def actions(agent_state, batch, key):
        return PyTreeDict(
            first=jnp.zeros((*batch.obs.first.shape[:-1], 2)),
            second=jnp.zeros((*batch.obs.second.shape[:-1], 1)),
        ), PyTreeDict()

    evaluator = MultiAgentEvaluator(
        env=env, action_fn=actions, max_episode_steps=5, discount=discount
    )
    evaluate = partial(evaluator.evaluate, None, num_episodes=5)
    key = jax.random.key(7)
    shape = ()
    if nested:
        evaluate = jax.vmap(jax.vmap(evaluate))
        key = jax.random.split(key, (2, 3))
        shape = (2, 3)
    result = (jax.jit(evaluate) if jit else evaluate)(key)
    assert result.episode_return.first.shape == (*shape, 6, num_objectives)
    assert result.episode_return.second.shape == (*shape, 6)
    assert result.episode_length.shape == (*shape, 6)
    expected = (1 + discount) * np.arange(1, num_objectives + 1)
    np.testing.assert_allclose(
        result.episode_return.first,
        np.broadcast_to(expected, (*shape, 6, num_objectives)),
    )
    np.testing.assert_allclose(
        result.episode_return.second, -2 * (1 + discount + discount**2)
    )
    np.testing.assert_array_equal(result.episode_length, 3)


@pytest.mark.parametrize("discount", [0.5, 1.0])
@pytest.mark.parametrize("episode_length", [3, 5], ids=["truncation", "termination"])
@pytest.mark.parametrize("nested", [False, True])
def test_returns_stop_at_agent_and_episode_boundaries(
    discount, episode_length, nested, jit
):
    env = VmapWrapper(
        OneEpisodeWrapper(ToyMultiAgentEnv(), episode_length), 2, vmap_step=True
    )

    def actions(agent_state, batch, key):
        return PyTreeDict(
            {
                name: jnp.argmax(obs.action_mask, axis=-1)
                for name, obs in batch.obs.items()
            }
        ), PyTreeDict()

    evaluator = MultiAgentEvaluator(
        env=env, action_fn=actions, max_episode_steps=6, discount=discount
    )
    evaluate = partial(evaluator.evaluate, None, num_episodes=5)
    key = jax.random.key(0)
    batch_shape = ()
    if nested:
        evaluate = jax.vmap(jax.vmap(evaluate))
        key = jax.random.split(key, (2, 3))
        batch_shape = (2, 3)
    result = (jax.jit(evaluate) if jit else evaluate)(key)

    # Five requested episodes round up to three batches of two environments.
    shape = (*batch_shape, 6)
    length = min(episode_length, 4)
    assert set(result.episode_return) == {"first", "second"}
    assert result.episode_length.shape == shape
    for returns in result.episode_return.values():
        assert returns.shape == shape
    np.testing.assert_allclose(result.episode_length, length)
    np.testing.assert_allclose(result.episode_return.first, 1 + discount)
    np.testing.assert_allclose(
        result.episode_return.second, -2 * sum(discount**t for t in range(length))
    )
