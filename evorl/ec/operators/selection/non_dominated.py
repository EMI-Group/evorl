import jax
import jax.numpy as jnp


def non_dominated_sort(fitnesses: jax.Array) -> jax.Array:
    """Assign zero-based Pareto ranks, maximizing every objective.

    Args:
        fitnesses: Objective values with shape `(population, objectives)`.

    Returns:
        Integer ranks with shape `(population,)`; zero is the Pareto front.
        Equal objective vectors do not dominate one another.
    """
    dominates = jnp.all(fitnesses[:, None] >= fitnesses[None, :], axis=-1)
    dominates &= jnp.any(fitnesses[:, None] > fitnesses[None, :], axis=-1)
    counts = dominates.sum(axis=0)
    ranks = jnp.full((fitnesses.shape[0],), -1, dtype=jnp.int32)

    def assign_front(carry):
        rank, ranks, counts = carry
        front = (counts == 0) & (ranks < 0)
        ranks = jnp.where(front, rank, ranks)
        counts -= (dominates & front[:, None]).sum(axis=0)
        return rank + 1, ranks, counts

    _, ranks, _ = jax.lax.while_loop(
        lambda carry: jnp.any(carry[1] < 0),
        assign_front,
        (jnp.int32(0), ranks, counts),
    )
    return ranks
