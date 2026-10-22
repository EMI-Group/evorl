from functools import partial
import operator
from typing import Literal

import jax
import jax.numpy as jnp


def non_dominated_sort(
    fitnesses: jax.Array,
    mode: Literal["auto", "blocked", "sweep", "ens_ss", "ens_bs"] = "auto",
    *,
    block_size: int = 128,
) -> jax.Array:
    """Assign zero-based Pareto ranks, maximizing at least two objectives.

    A point dominates another when it is at least as good in every objective
    and strictly better in at least one. Equal objective vectors have equal
    ranks. Rank zero is the nondominated front; removing it exposes rank one,
    and so on.

    All modes support JIT and nested `vmap`. Mark `mode` and `block_size` static
    when passing them as JIT arguments. Changing either creates a separately
    compiled variant. Automatic backend selection uses the compilation platform,
    including when inputs are explicitly placed on a CPU while the default
    backend is a GPU. No host callback is used.

    Args:
        fitnesses: Real objective values with shape `(population, objectives)`
            and at least two objective columns. NaNs are not supported. An
            empty population returns an empty rank array.
        mode: Static algorithm choice. `auto` uses `sweep` for two objectives
            on CPU, `ens_bs` for more objectives on CPU, and `blocked` on other
            backends. `sweep` is a two-objective scan with binary searches over
            front boundaries. `ens_ss` and `ens_bs` use Efficient Nondominated
            Sorting with sequential or binary front search. `blocked` processes
            ready points in bounded batches without retaining a full dominance
            matrix; it supports any objective count of at least two.
        block_size: Positive static integer limiting the number of ready points
            processed per batch. Only used by `blocked`, including when `auto`
            selects it; ignored by sweep and ENS. The effective size is capped
            at the population size. Arbitrary positive integers are supported,
            not only powers of two. Larger blocks use more scratch memory and
            may reduce batch count, but can waste work when few points are ready.
            The default 128 is a memory/performance compromise, not a hardware
            workgroup size or a universally optimal value.

    Returns:
        An `int32` array of shape `(population,)` with the Pareto ranks in the
        original input order.

    Raises:
        TypeError: If `block_size` cannot be interpreted as an integer.
        ValueError: If the input is not two-dimensional, has fewer than two
            objectives, the mode is unknown, or `sweep` receives more than two
            objectives, or `block_size` is not positive.
    """
    fitnesses = jnp.asarray(fitnesses)
    if fitnesses.ndim != 2 or fitnesses.shape[1] < 2:
        raise ValueError("fitnesses must have shape (population, objectives >= 2)")
    if mode not in ("auto", "blocked", "sweep", "ens_ss", "ens_bs"):
        raise ValueError(f"Unknown nondominated sorting mode: {mode!r}")
    if mode == "sweep" and fitnesses.shape[1] != 2:
        raise ValueError("sweep requires exactly two objectives")
    block_size = operator.index(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if fitnesses.shape[0] == 0:
        return jnp.empty((0,), dtype=jnp.int32)

    if mode == "auto":
        cpu_sort = (
            _sweep_sort if fitnesses.shape[1] == 2 else partial(_ens_sort, binary=True)
        )
        return jax.lax.platform_dependent(
            fitnesses,
            cpu=cpu_sort,
            default=partial(_blocked_sort, block_size=block_size),
        )
    if mode == "blocked":
        return _blocked_sort(fitnesses, block_size=block_size)
    if mode == "sweep":
        return _sweep_sort(fitnesses)
    return _ens_sort(fitnesses, binary=mode == "ens_bs")


def _blocked_sort(fitnesses: jax.Array, *, block_size: int) -> jax.Array:
    """Rank points by processing their dominance dependencies in bounded batches.

    First count incoming dominance edges in blocks. Then process up to
    `block_size` points whose dominators have all been processed, recomputing those
    outgoing rows. Propagate the maximum predecessor rank plus one to each
    dominated point. No edges remain when every rank is final; the last ready
    points need no further processing.

    With `N` points, `M` objectives, block size `B`, and `T` batches, comparison
    work is `O(M*N**2 + M*B*N*T)`. For fixed `M`, scratch space is `O(B*N)`.
    Batches with few ready points still incur work for masked padding slots.

    Args:
        fitnesses: Nonempty objective array of shape `(population, objectives)`.
        block_size: Positive static integer bounding the comparison block height.

    Returns:
        The Pareto ranks as an `int32` array of shape `(population,)`.
    """
    size = fitnesses.shape[0]
    block_size = min(block_size, size)
    offsets = jnp.arange(block_size, dtype=jnp.int32)

    def outgoing(indices):
        sources = fitnesses[jnp.minimum(indices, size - 1)]
        dominates = jnp.all(sources[:, None] >= fitnesses[None], axis=-1)
        dominates &= jnp.any(sources[:, None] > fitnesses[None], axis=-1)
        return dominates & (indices < size)[:, None]

    def count_incoming(block, counts):
        edges = outgoing(block * block_size + offsets)
        return counts + edges.sum(axis=0, dtype=jnp.int32)

    counts = jax.lax.fori_loop(
        0,
        (size + block_size - 1) // block_size,
        count_incoming,
        jnp.zeros(size, dtype=jnp.int32),
    )

    def process(carry):
        counts, ranks, processed = carry
        indices = jnp.nonzero(
            (counts == 0) & ~processed, size=block_size, fill_value=size
        )[0]
        edges = outgoing(indices)
        source_ranks = ranks[jnp.minimum(indices, size - 1)]
        updates = jnp.max(jnp.where(edges, source_ranks[:, None] + 1, 0), axis=0)
        return (
            counts - edges.sum(axis=0, dtype=jnp.int32),
            jnp.maximum(ranks, updates),
            processed.at[indices].set(True, mode="drop"),
        )

    _, ranks, _ = jax.lax.while_loop(
        lambda carry: jnp.any(carry[0] > 0),
        process,
        (counts, jnp.zeros(size, jnp.int32), jnp.zeros(size, jnp.bool_)),
    )
    return ranks


def _sweep_sort(fitnesses: jax.Array) -> jax.Array:
    """Rank two-objective points by scanning decreasing lexicographic order.

    Front boundaries store the largest second-objective value in each front.
    They decrease with the front rank, so binary search locates the first
    front that cannot dominate the current point. Identical adjacent points
    reuse their rank. Work is `O(N*log(N))` and scratch space is `O(N)`.

    Args:
        fitnesses: Nonempty objective array of shape `(population, 2)`.

    Returns:
        The Pareto ranks as an `int32` array of shape `(population,)`.
    """
    size = fitnesses.shape[0]
    # Reverse ascending order instead of negating possibly minimal integers.
    order = jnp.lexsort(fitnesses.T[::-1])[::-1]
    values = fitnesses[order]

    def process(carry, index):
        boundaries, num_fronts, previous_rank = carry
        point = values[index]
        duplicate = (index > 0) & jnp.all(point == values[jnp.maximum(index - 1, 0)])

        def search(_):
            def bisect(bounds):
                low, high = bounds
                middle = (low + high) // 2
                dominated = boundaries[middle] >= point[1]
                return (
                    jnp.where(dominated, middle + 1, low),
                    jnp.where(dominated, high, middle),
                )

            return jax.lax.while_loop(
                lambda bounds: bounds[0] < bounds[1],
                bisect,
                (jnp.int32(0), num_fronts),
            )[0]

        rank = jax.lax.cond(duplicate, lambda _: previous_rank, search, None)
        return (
            boundaries.at[rank].set(point[1]),
            jnp.maximum(num_fronts, rank + 1),
            rank,
        ), rank

    _, ranks = jax.lax.scan(
        process,
        (jnp.zeros(size, fitnesses.dtype), jnp.int32(0), jnp.int32(0)),
        jnp.arange(size, dtype=jnp.int32),
    )
    return jnp.zeros(size, dtype=jnp.int32).at[order].set(ranks)


def _ens_sort(fitnesses: jax.Array, *, binary: bool) -> jax.Array:
    """Rank lexicographically ordered points using Efficient Nondominated Sorting.

    Each front is a linked list stored in fixed-size arrays. A front check
    stops at the first dominator. Binary search skips fronts when `binary`
    is true; otherwise fronts are checked sequentially. Both variants have
    worst-case `O(M*N**2)` comparison work and `O(N)` auxiliary indexing space,
    plus storage for reordered objectives. Duplicate points share one list node.

    Reference: https://doi.org/10.1109/TEVC.2014.2308305.

    Args:
        fitnesses: Nonempty objective array of shape `(population, objectives)`.
        binary: Static choice of binary rather than sequential front search.

    Returns:
        The Pareto ranks as an `int32` array of shape `(population,)`.
    """
    size = fitnesses.shape[0]
    order = jnp.lexsort(fitnesses.T[::-1])[::-1]
    values = fitnesses[order]

    def process(carry, index):
        heads, next_node, num_fronts, previous_rank = carry
        point = values[index]
        duplicate = (index > 0) & jnp.all(point == values[jnp.maximum(index - 1, 0)])

        def assign(_):
            def dominated(front):
                def visit(state):
                    node, _ = state
                    other = values[node]
                    found = jnp.all(other >= point) & jnp.any(other > point)
                    return next_node[node], found

                return jax.lax.while_loop(
                    lambda state: (state[0] >= 0) & ~state[1],
                    visit,
                    (heads[front], jnp.bool_(False)),
                )[1]

            if binary:

                def bisect(bounds):
                    low, high = bounds
                    middle = (low + high) // 2
                    hit = dominated(middle)
                    return (
                        jnp.where(hit, middle + 1, low),
                        jnp.where(hit, high, middle),
                    )

                rank = jax.lax.while_loop(
                    lambda bounds: bounds[0] < bounds[1],
                    bisect,
                    (jnp.int32(0), num_fronts),
                )[0]
            else:

                def advance(state):
                    front, _ = state
                    hit = dominated(front)
                    return front + hit.astype(jnp.int32), hit

                rank = jax.lax.while_loop(
                    lambda state: (state[0] < num_fronts) & state[1],
                    advance,
                    (jnp.int32(0), jnp.bool_(True)),
                )[0]

            return (
                heads.at[rank].set(index),
                next_node.at[index].set(heads[rank]),
                jnp.maximum(num_fronts, rank + 1),
                rank,
            )

        carry = jax.lax.cond(duplicate, lambda _: carry, assign, None)
        return carry, carry[3]

    _, ranks = jax.lax.scan(
        process,
        (
            jnp.full(size, -1, dtype=jnp.int32),
            jnp.full(size, -1, dtype=jnp.int32),
            jnp.int32(0),
            jnp.int32(0),
        ),
        jnp.arange(size, dtype=jnp.int32),
    )
    return jnp.zeros(size, dtype=jnp.int32).at[order].set(ranks)


# Previous dense implementation, retained for reference. It stores an N x N
# dominance matrix and rescans every row for each Pareto front: O(N**2) scratch
# space and O(M*N**2 + F*N**2) work for M objectives and F fronts, up to O(N**3)
# work for fixed M. The implementations above avoid retaining this matrix.
# def non_dominated_sort(fitnesses: jax.Array) -> jax.Array:
#     """Assign zero-based Pareto ranks, maximizing every objective.
#
#     Args:
#         fitnesses: Objective values with shape `(population, objectives)`.
#
#     Returns:
#         Integer ranks with shape `(population,)`; zero is the Pareto front.
#         Equal objective vectors do not dominate one another.
#     """
#     dominates = jnp.all(fitnesses[:, None] >= fitnesses[None, :], axis=-1)
#     dominates &= jnp.any(fitnesses[:, None] > fitnesses[None, :], axis=-1)
#     counts = dominates.sum(axis=0)
#     ranks = jnp.full((fitnesses.shape[0],), -1, dtype=jnp.int32)
#
#     def assign_front(carry):
#         rank, ranks, counts = carry
#         front = (counts == 0) & (ranks < 0)
#         ranks = jnp.where(front, rank, ranks)
#         counts -= (dominates & front[:, None]).sum(axis=0)
#         return rank + 1, ranks, counts
#
#     _, ranks, _ = jax.lax.while_loop(
#         lambda carry: jnp.any(carry[1] < 0),
#         assign_front,
#         (jnp.int32(0), ranks, counts),
#     )
#     return ranks
