"""Check independent rings through the real off-policy shard_map boundary."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evorl.agent import AgentState
from evorl.replay_buffers import (
    LAPReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from evorl.types import PyTreeDict
from evorl.workflows import OffPolicyWorkflow

from tests.helpers.sharding import compiled_collectives

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "buffer_type",
    [ReplayBuffer, PrioritizedReplayBuffer, LAPReplayBuffer],
    ids=["uniform", "prioritized", "lap"],
)
def test_independent_masked_rings(buffer_type, jit, dp_mesh):
    mesh = dp_mesh
    buffer = buffer_type(capacity=4, sample_batch_size=3)
    prioritized = isinstance(buffer, PrioritizedReplayBuffer)

    class Env:
        def reset(self, key):
            rank = jax.lax.axis_index("DP")
            return PyTreeDict(
                obs=rank * 100 + jnp.arange(3, dtype=jnp.float32),
                mask=jnp.where(
                    rank == 0,
                    jnp.array([False, True, False]),
                    jnp.array([True, False, True]),
                ),
            )

    class Harness(OffPolicyWorkflow):
        def __init__(self):
            self.dp_axis_name = "DP"
            self.mesh = mesh
            self.env = Env()
            self.replay_buffer = buffer
            self.config = SimpleNamespace()

        def _setup_agent_and_optimizer(self, key):
            return AgentState(params=jnp.zeros(1)), jnp.zeros(1)

        def _setup_replaybuffer(self, key):
            # Defined unused entries let us compare the entire ring after writes.
            return buffer.init(jnp.float32(0)).replace(data=jnp.full(4, -1.0))

        def _postsetup_replaybuffer(self, state):
            return state.replace(
                replay_buffer_state=buffer.add(
                    state.replay_buffer_state, state.env_state.obs, state.env_state.mask
                )
            )

        def step(self, state):
            key, sample_key = jax.random.split(state.key)
            rank = jax.lax.axis_index(self.dp_axis_name)
            sample_key = jax.random.fold_in(sample_key, rank)
            iteration = state.metrics.iterations
            count = jnp.where(rank == 0, (iteration + 1) % 4, iteration % 3 + 1)
            values = state.env_state.obs + 3
            rb = buffer.add(state.replay_buffer_state, values, jnp.arange(3) < count)
            if prioritized:
                samples, _, rb = buffer.sample(rb, sample_key)
                rb = buffer.update_priority(rb, jnp.abs(samples) + 1)
                rb = buffer.reset_max_priority(rb)
            else:
                samples = buffer.sample(rb, sample_key)
            # Retain samples in a sharded field to verify their ownership.
            return jnp.float32(0), state.replace(
                key=key,
                replay_buffer_state=rb,
                env_state=state.env_state.replace(obs=values, mask=samples),
                metrics=state.metrics.replace(iterations=iteration + 1),
            )

        def evaluate(self, state):
            return jnp.float32(0), state

    with jax.set_mesh(mesh):
        Harness.enable_shmap()
        if jit:
            Harness.enable_jit()
        workflow = Harness()
        state = workflow.setup(jax.random.key(42))
        assert state.key.shape == () and state.key.sharding.is_fully_replicated
        assert state.replay_buffer_state.data.shape == (8,)
        assert state.replay_buffer_state.current_index.shape == (2,)
        np.testing.assert_array_equal(state.replay_buffer_state.buffer_size, [1, 2])
        reference = []
        rings = []
        pointers = [0, 0]
        sizes = [0, 0]

        def write_reference(rank, values, mask):
            for value in np.asarray(values)[np.asarray(mask)]:
                rings[rank][pointers[rank]] = value
                pointers[rank] = (pointers[rank] + 1) % 4
                sizes[rank] = min(sizes[rank] + 1, 4)

        for rank in range(2):
            values = rank * 100 + jnp.arange(3, dtype=jnp.float32)
            mask = jnp.array([False, True, False] if rank == 0 else [True, False, True])
            rb = buffer.init(jnp.float32(0)).replace(data=jnp.full(4, -1.0))
            reference.append(buffer.add(rb, values, mask))
            rings.append(np.full(4, -1.0))
            write_reference(rank, values, mask)

        if jit:
            hlo = Harness.step.lower(workflow, state).compile().as_text()
            assert not compiled_collectives(hlo)

        root_key = state.key
        for iteration in range(6):
            next_key, sample_key = jax.random.split(root_key)
            _, state = workflow.step(state)
            _, state = workflow.evaluate(state)
            rb = state.replay_buffer_state
            assert rb.data.shape == (8,)
            for rank in range(2):
                local_key = jax.random.fold_in(sample_key, rank)
                count = (iteration + 1) % 4 if rank == 0 else iteration % 3 + 1
                values = (
                    rank * 100 + (iteration + 1) * 3 + jnp.arange(3, dtype=jnp.float32)
                )
                mask = jnp.arange(3) < count
                expected = buffer.add(reference[rank], values, mask)
                write_reference(rank, values, mask)
                if prioritized:
                    samples, _, expected = buffer.sample(expected, local_key)
                    expected = buffer.update_priority(expected, jnp.abs(samples) + 1)
                    expected = buffer.reset_max_priority(expected)
                else:
                    samples = buffer.sample(expected, local_key)
                reference[rank] = expected
                # Buffer data and sampled indices remain local, including after
                # unequal masked writes, zero writes, saturation and wraparound.
                actual_data = rb.data.addressable_shards[rank].data
                np.testing.assert_array_equal(actual_data, rings[rank])
                for actual, expected_value in zip(
                    jax.tree.leaves(rb), jax.tree.leaves(expected), strict=True
                ):
                    actual_value = np.asarray(actual.addressable_shards[rank].data)
                    if expected_value.ndim == 0:
                        actual_value = actual_value[0]
                    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6)
                actual_samples = state.env_state.mask.addressable_shards[rank].data
                np.testing.assert_array_equal(actual_samples, samples)
                assert np.all(np.asarray(actual_samples) >= rank * 100)
                assert np.all(np.asarray(actual_samples) < (rank + 1) * 100)
                np.testing.assert_array_equal(
                    jax.random.key_data(state.key.addressable_shards[rank].data),
                    jax.random.key_data(next_key),
                )
            np.testing.assert_array_equal(rb.current_index, pointers)
            np.testing.assert_array_equal(rb.buffer_size, sizes)
            root_key = next_key
        if prioritized:
            assert rb.priority.shape == (8,)
            assert rb.sample_indices.shape == (6,)
            assert rb.max_priority.shape == (2,)
            assert np.asarray(rb.max_priority)[0] != np.asarray(rb.max_priority)[1]
