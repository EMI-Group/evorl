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
    assert set(result.episode_returns) == {"first", "second"}
    assert result.episode_lengths.shape == shape
    for returns in result.episode_returns.values():
        assert returns.shape == shape
    np.testing.assert_allclose(result.episode_lengths, length)
    np.testing.assert_allclose(result.episode_returns.first, 1 + discount)
    np.testing.assert_allclose(
        result.episode_returns.second, -2 * sum(discount**t for t in range(length))
    )
