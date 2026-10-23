"""Replay sampling ignores unwritten storage, including NaNs and masked writes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer, LAPReplayBuffer
from evorl.types import PyTreeDict


@pytest.mark.parametrize(
    "buffer_type", [ReplayBuffer, PrioritizedReplayBuffer, LAPReplayBuffer]
)
@pytest.mark.parametrize("masked", [False, True])
def test_sampling_uses_written_entries_before_and_after_ring_wrap(
    buffer_type, masked, jit
):
    buffer = buffer_type(capacity=8, sample_batch_size=32, min_sample_timesteps=2)
    spec = PyTreeDict(obs=jnp.zeros(3), rewards=jnp.zeros(2), done=jnp.bool_(False))
    init = jax.jit(buffer.init) if jit else buffer.init
    add = jax.jit(buffer.add) if jit else buffer.add
    sample = jax.jit(buffer.sample) if jit else buffer.sample

    # Unwritten storage is intentionally undefined. Disable NaN debugging here
    # so it does not reject the poisoned storage before sampling.
    # The sampled data is checked explicitly below.
    with jax.debug_nans(False):
        state = init(spec)
        assert int(state.buffer_size) == 0
        assert not bool(buffer.can_sample(state))
        state = state.replace(
            data=jax.tree.map(
                lambda x: (
                    jnp.full_like(x, jnp.nan)
                    if jnp.issubdtype(x.dtype, jnp.floating)
                    else x
                ),
                state.data,
            )
        )
        expected = np.full(8, np.nan)
        pointer = 0

        for index, values in enumerate((np.arange(10, 14), np.arange(20, 27))):
            batch = PyTreeDict(
                obs=jnp.broadcast_to(
                    jnp.asarray(values[:, None], jnp.float32), (len(values), 3)
                ),
                rewards=jnp.asarray(np.stack((values, -values), axis=-1), jnp.float32),
                done=jnp.asarray(values % 2 == 0),
            )
            mask = (
                np.arange(len(values)) != 1
                if masked
                else np.ones(len(values), dtype=bool)
            )
            state = add(state, batch, jnp.asarray(mask) if masked else None)
            for value in values[mask]:
                expected[pointer] = value
                pointer = (pointer + 1) % buffer.capacity
            count = int(np.isfinite(expected).sum())
            assert int(state.buffer_size) == count
            assert int(state.current_index) == pointer
            assert bool(buffer.can_sample(state))
            np.testing.assert_array_equal(state.data.obs[:, 0], expected)
            if index == 0:
                assert np.isnan(np.asarray(state.data.rewards)[count:]).all()

            result = sample(state, jax.random.key(index))
            if buffer_type is ReplayBuffer:
                samples = result
            else:
                samples, weights, _ = result
                assert np.isfinite(weights).all()
                assert np.all(np.asarray(weights) > 0)
            assert all(np.isfinite(x).all() for x in jax.tree.leaves(samples))
            ids = np.asarray(samples.obs)[:, 0]
            assert np.isin(ids, expected[np.isfinite(expected)]).all()
            np.testing.assert_array_equal(
                samples.obs, np.broadcast_to(ids[:, None], (32, 3))
            )
            np.testing.assert_array_equal(
                samples.rewards, np.stack((ids, -ids), axis=-1)
            )
            np.testing.assert_array_equal(samples.done, ids % 2 == 0)
