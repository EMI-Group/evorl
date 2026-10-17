"""Compare nested early-stop evaluation with independent episode loops."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.envs.gymnasium import OneEpisodeWrapper
from evorl.rollout import fast_eval_rollout_episode, host_fast_eval_rollout_episode
from evorl.types import PyTreeDict
from tests.helpers.envs import ToyEnv


def test_nested_evaluation_preserves_completed_members():
    class VariableLengthEnv(ToyEnv):
        def reset(self, key):
            state = super().reset(key)
            lengths = jax.random.randint(key, (self.num_envs,), 1, 5)
            return state.replace(info=state.info.replace(lengths=lengths))

        def step(self, state, action):
            next_state = super().step(state, action)
            return next_state.replace(
                done=(next_state.env_state >= state.info.lengths).astype(jnp.float32)
            )

    env = VariableLengthEnv(2)

    def action_fn(agent_state, batch, key):
        return batch.obs[:, :1] * agent_state, PyTreeDict()

    def evaluate(key, agent_state):
        return fast_eval_rollout_episode(
            env.step, action_fn, env.reset(key), agent_state, key, 8
        )

    keys = jax.random.split(jax.random.key(2), (2, 3))
    # The shared scalar exercises broadcasting of unbatched agent leaves.
    nested = jax.jit(jax.vmap(jax.vmap(evaluate, in_axes=(0, None)), in_axes=(0, None)))
    actual = nested(keys, jnp.float32(0.5))
    independent = jax.jit(evaluate)
    expected = [independent(key, jnp.float32(0.5)) for key in keys.reshape(-1)]
    expected = jax.tree.map(
        lambda *xs: jnp.stack(xs).reshape((2, 3, *xs[0].shape)), *expected
    )
    for result, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)
    metrics, state = actual
    assert np.unique(np.asarray(metrics.episode_lengths.max(-1))).size > 1
    # Completed members freeze at their own stopping point while others run.
    assert np.all(np.asarray(state.env_state) <= 4)


@pytest.mark.parametrize("shared_key", [False, True], ids=["mapped_key", "shared_key"])
@pytest.mark.parametrize("rollout_length", [2, 8], ids=["step_limit", "early_stop"])
def test_host_evaluation_with_mixed_agent_axes(shared_key, rollout_length):
    class VariableLengthEnv(ToyEnv):
        def reset(self, key):
            state = super().reset(key)
            lengths = jax.random.randint(key, (self.num_envs,), 1, 5)
            return state.replace(info=state.info.replace(lengths=lengths))

        def step(self, state, action):
            state = super().step(state, action)
            return state.replace(
                done=(state.env_state >= state.info.lengths).astype(jnp.float32)
            )

    # Host evaluation relies on the adapter's existing per-environment freezing.
    env = OneEpisodeWrapper(VariableLengthEnv(2))

    def action_fn(agent_state, batch, key):
        actions = batch.obs[:, :1] @ agent_state.weight + agent_state.bias
        return actions + jax.random.uniform(key, actions.shape), PyTreeDict()

    def evaluate(state, agent_state, key):
        return host_fast_eval_rollout_episode(
            env.step, action_fn, state, agent_state, key, rollout_length
        )

    keys = jax.random.split(jax.random.key(2), (2, 3))
    states = jax.jit(jax.vmap(jax.vmap(env.reset)))(keys)
    agent_state = PyTreeDict(
        weight=jnp.array([[0.5]]), bias=jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    )
    axes = (0, PyTreeDict(weight=None, bias=0), None if shared_key else 0)
    eval_keys = jax.random.key(3) if shared_key else keys
    nested = jax.jit(jax.vmap(jax.vmap(evaluate, in_axes=axes), in_axes=axes))
    actual = nested(states, agent_state, eval_keys)

    reference = jax.jit(
        lambda state, agent_state, key: fast_eval_rollout_episode(
            env.step, action_fn, state, agent_state, key, rollout_length
        )
    )
    expected = [
        reference(
            jax.tree.map(lambda x, i=i, j=j: x[i, j], states),
            agent_state.replace(bias=agent_state.bias[i, j]),
            eval_keys if shared_key else eval_keys[i, j],
        )
        for i in range(2)
        for j in range(3)
    ]
    expected = jax.tree.map(
        lambda *xs: jnp.stack(xs).reshape((2, 3, *xs[0].shape)), *expected
    )
    for result, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        actual[0].episode_lengths, np.minimum(states.info.lengths, rollout_length)
    )
