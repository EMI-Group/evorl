"""Pareto sorting correctness, execution modes and bounded scratch memory."""

from functools import partial
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.ec.operators.selection import non_dominated as nds
from tests.helpers.sharding import compiled_collectives

MODES = ["auto", "blocked", "sweep", "ens_ss", "ens_bs"]
CASES = [
    (mode, objectives)
    for mode in MODES
    for objectives in [2, 3, 5]
    if mode != "sweep" or objectives == 2
]


def reference_ranks(fitnesses):
    """Peel fronts in NumPy independently of the JAX algorithms."""
    remaining = list(range(len(fitnesses)))
    ranks = np.full(len(fitnesses), -1, dtype=np.int32)
    rank = 0
    while remaining:
        front = [
            i
            for i in remaining
            if not any(
                np.all(fitnesses[j] >= fitnesses[i])
                and np.any(fitnesses[j] > fitnesses[i])
                for j in remaining
            )
        ]
        ranks[front] = rank
        remaining = [i for i in remaining if i not in front]
        rank += 1
    return ranks


@pytest.mark.parametrize("mode,objectives", CASES)
def test_modes_match_independent_front_peeling(mode, objectives, jit):
    fitnesses = np.random.default_rng(42).integers(-3, 4, (19, objectives))
    sort = (
        jax.jit(nds.non_dominated_sort, static_argnames=("mode",))
        if jit
        else nds.non_dominated_sort
    )
    actual = sort(jnp.asarray(fitnesses), mode=mode)
    assert actual.dtype == jnp.int32
    np.testing.assert_array_equal(actual, reference_ranks(fitnesses))


@pytest.mark.parametrize("mode", MODES)
def test_known_fronts_ties_duplicates_and_infinities(mode):
    # Same population size, with very different front depths and duplicate sets.
    chain = np.repeat(np.arange(7, dtype=np.float32)[:, None], 2, axis=1)
    antichain = chain * np.array([1, -1], np.float32)
    identical = np.ones((7, 2), np.float32)
    ties = np.array(
        [[3, 3], [3, 3], [3, 2], [2, 3], [2, 2], [1, 3], [0, 0]], np.float32
    )
    infinities = np.array(
        [
            [np.inf, np.inf],
            [np.inf, 0],
            [0, np.inf],
            [0, 0],
            [-np.inf, 0],
            [0, -np.inf],
            [-np.inf, -np.inf],
        ],
        np.float32,
    )
    batches = np.stack((chain, antichain, identical, ties, infinities))
    sort = partial(nds.non_dominated_sort, mode=mode)
    actual = jax.jit(jax.vmap(sort))(jnp.asarray(batches))
    np.testing.assert_array_equal(
        actual, np.stack([reference_ranks(x) for x in batches])
    )


@pytest.mark.parametrize("mode", MODES)
def test_sorting_does_not_negate_integer_objectives(mode):
    bounds = np.iinfo(np.int32)
    fitnesses = jnp.array(
        [[bounds.min, 0], [bounds.max, 0], [0, 0], [bounds.max, 0], [0, bounds.min]],
        jnp.int32,
    )
    sort = jax.jit(nds.non_dominated_sort, static_argnames=("mode",))
    np.testing.assert_array_equal(sort(fitnesses, mode=mode), [2, 0, 1, 0, 2])


@pytest.mark.parametrize(
    "mode,objectives", [(mode, 2 if mode == "sweep" else 3) for mode in MODES]
)
def test_nested_vmap_with_different_front_depths(mode, objectives):
    rng = np.random.default_rng(3)
    batches = rng.integers(-3, 4, (2, 2, 9, objectives))
    batches[0, 0] = np.arange(9)[:, None]
    batches[0, 1] = 1
    batches[1, 0, :, 0] = np.arange(9)
    batches[1, 0, :, 1] = -np.arange(9)
    sort = partial(nds.non_dominated_sort, mode=mode, block_size=3)
    actual = jax.jit(jax.vmap(jax.vmap(sort)))(jnp.asarray(batches))
    expected = np.stack(
        [np.stack([reference_ranks(x) for x in row]) for row in batches]
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("size", [127, 129, 257])
def test_blocked_sort_handles_partial_blocks_and_ready_batches(size):
    # More than 128 ready points for size=257; the last input block is partial.
    levels = (np.arange(size) + 1) % 2
    fitnesses = np.repeat(levels[:, None], 3, axis=1)
    order = np.random.default_rng(5).permutation(size)
    sort = jax.jit(partial(nds.non_dominated_sort, mode="blocked"))
    np.testing.assert_array_equal(
        sort(jnp.asarray(fitnesses[order])), 1 - levels[order]
    )


@pytest.mark.parametrize("block_size", [1, 3, 16, 64, 128, 256])
def test_block_size_and_input_order_preserve_duplicate_and_tied_ranks(block_size, jit):
    base = np.array(
        [[3, 3], [3, 3], [3, 2], [2, 3], [2, 2], [2, 2], [1, 3], [0, 0]],
        np.float32,
    )
    fitnesses = np.tile(base, (17, 1))
    expected = np.tile(reference_ranks(base), 17)
    permutation = np.random.default_rng(13).permutation(len(fitnesses))
    sort = (
        jax.jit(nds.non_dominated_sort, static_argnames=("mode", "block_size"))
        if jit
        else nds.non_dominated_sort
    )
    for order in [np.arange(len(fitnesses)), permutation]:
        actual = sort(
            jnp.asarray(fitnesses[order]), mode="blocked", block_size=block_size
        )
        np.testing.assert_array_equal(actual, expected[order])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("block_size", [0, -1])
def test_rejects_nonpositive_block_sizes(mode, block_size):
    sort = jax.jit(nds.non_dominated_sort, static_argnames=("mode", "block_size"))
    with pytest.raises(ValueError, match="block_size must be positive"):
        sort(jnp.ones((0, 2)), mode=mode, block_size=block_size)


def test_block_size_requires_a_static_integer():
    sort = jax.jit(nds.non_dominated_sort, static_argnames=("mode", "block_size"))
    with pytest.raises(TypeError, match="integer"):
        sort(jnp.ones((4, 2)), mode="blocked", block_size=1.5)
    np.testing.assert_array_equal(
        sort(jnp.ones((4, 2)), mode="blocked", block_size=np.int64(3)),
        np.zeros(4, np.int32),
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("size", [0, 1])
def test_empty_and_singleton_populations(mode, size):
    sort = jax.jit(nds.non_dominated_sort, static_argnames=("mode",))
    ranks = sort(jnp.ones((size, 2)), mode=mode)
    np.testing.assert_array_equal(ranks, np.zeros(size, dtype=np.int32))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", [(4,), (4, 0), (4, 1), (2, 3, 2)])
def test_rejects_invalid_shapes_and_fewer_than_two_objectives(mode, shape):
    sort = jax.jit(nds.non_dominated_sort, static_argnames=("mode",))
    with pytest.raises(ValueError, match="objectives >= 2"):
        sort(jnp.zeros(shape), mode=mode)


def test_rejects_unknown_modes_and_sweep_with_more_than_two_objectives():
    with pytest.raises(ValueError, match="Unknown"):
        nds.non_dominated_sort(jnp.ones((4, 2)), mode="other")
    with pytest.raises(ValueError, match="exactly two"):
        nds.non_dominated_sort(jnp.ones((4, 3)), mode="sweep")


@pytest.mark.parametrize("objectives", [2, 3])
def test_auto_uses_actual_execution_backend(objectives):
    # On GPU runs this also exercises explicitly placed CPU inputs while CUDA
    # remains the default backend. Keep a real data dependency: constant-only
    # branches let JAX discard the input and choose the default output device.
    devices = [jax.devices("cpu")[0]]
    if jax.default_backend() != "cpu":
        devices.append(jax.devices()[0])
    with (
        patch.object(nds, "_sweep_sort", lambda x: x[:, 0].astype(jnp.int32)),
        patch.object(
            nds, "_ens_sort", lambda x, *, binary: x[:, 0].astype(jnp.int32) + 1
        ),
        patch.object(
            nds,
            "_blocked_sort",
            lambda x, *, block_size: x[:, 0].astype(jnp.int32) + block_size,
        ),
    ):
        sort = jax.jit(lambda x: nds.non_dominated_sort(x, block_size=17))
        for device in devices:
            fitnesses = jax.device_put(np.ones((4, objectives), np.float32), device)
            expected = (1 if objectives == 2 else 2) if device.platform == "cpu" else 18
            np.testing.assert_array_equal(sort(fitnesses), np.full(4, expected))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("nested", [False, True], ids=["single", "nested_vmap"])
def test_replicated_sort_has_no_collectives(mode, nested, mesh_factory):
    mesh = mesh_factory(shape=(min(2, jax.local_device_count()),))
    fitnesses = np.array([[3, 3], [3, 2], [2, 3], [2, 2], [2, 2], [0, 0]])
    sort_fn = partial(nds.non_dominated_sort, mode=mode, block_size=3)
    if nested:
        chain = np.repeat(np.arange(6)[:, None], 2, axis=1)
        antichain = chain * np.array([1, -1])
        identical = np.ones_like(chain)
        fitnesses = np.stack((fitnesses, chain, antichain, identical)).reshape(
            2, 2, 6, 2
        )
        expected = np.stack(
            [np.stack([reference_ranks(x) for x in row]) for row in fitnesses]
        )
        sort_fn = jax.vmap(jax.vmap(sort_fn))
    else:
        expected = reference_ranks(fitnesses)
    with jax.set_mesh(mesh):
        fitnesses = jax.device_put(fitnesses, jax.P())
        sort = jax.jit(
            jax.shard_map(
                sort_fn,
                in_specs=jax.P(),
                out_specs=jax.P(),
            )
        )
        executable = sort.lower(fitnesses).compile()
        np.testing.assert_array_equal(executable(fitnesses), expected)
    assert not compiled_collectives(executable.as_text())


def test_blocked_scratch_memory_grows_linearly():
    sort = jax.jit(partial(nds.non_dominated_sort, mode="blocked"))
    sizes = [4096, 16384]
    memory = [
        sort.lower(jax.ShapeDtypeStruct((size, 3), jnp.float32))
        .compile()
        .memory_analysis()
        .temp_size_in_bytes
        for size in sizes
    ]
    # Quadrupling N must not require a quadratic (16x) scratch buffer.
    assert memory[1] < 6 * memory[0]
