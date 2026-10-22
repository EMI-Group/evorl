import math

import jax
import jax.numpy as jnp

from evorl.ec.operators.selection.non_dominated import non_dominated_sort
from evorl.types import Params, PyTreeData, PyTreeDict, pytree_field
from evorl.utils.ec_utils import ParamVectorSpec

from .ec_optimizer import EvoOptimizer


class NSGA2State(PyTreeData):
    """Evaluated parents and the pending population returned by `ask`."""

    population: jax.Array
    fitnesses: jax.Array
    ranks: jax.Array
    crowding: jax.Array
    offspring: jax.Array
    initialized: jax.Array
    key: jax.Array


class NSGA2(EvoOptimizer):
    """NSGA-II with bounded SBX crossover and polynomial mutation.

    At least two objectives are required and every objective is maximized.
    `ask` first returns a uniformly initialized population, then offspring
    selected by rank and crowding tournaments.
    `tell` keeps the best `pop_size` parents and offspring together.

    Reference: Deb et al., "A fast and elitist multiobjective genetic
    algorithm: NSGA-II", https://doi.org/10.1109/4235.996017.
    """

    pop_size: int
    num_objectives: int
    param_vec_spec: ParamVectorSpec = pytree_field(static=True)
    lb: float = -10.0
    ub: float = 10.0
    crossover_probability: float = 0.9
    mutation_probability: float | None = None
    crossover_eta: float = 20.0
    mutation_eta: float = 20.0

    def __post_init__(self):
        if self.pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if self.num_objectives < 2:
            raise ValueError("num_objectives must be at least two")
        if (
            not math.isfinite(self.lb)
            or not math.isfinite(self.ub)
            or self.lb >= self.ub
        ):
            raise ValueError("lb and ub must be finite with lb < ub")
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("crossover_probability must be in [0, 1]")
        if self.mutation_probability is None:
            self.mutation_probability = 1.0 / self.param_vec_spec.vec_size
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("mutation_probability must be in [0, 1]")
        if self.crossover_eta <= 0 or self.mutation_eta <= 0:
            raise ValueError(
                "Crossover and mutation distribution indices must be positive"
            )

    def init(self, key: jax.Array) -> NSGA2State:
        key, sample_key = jax.random.split(key)
        population = jax.random.uniform(
            sample_key,
            (self.pop_size, self.param_vec_spec.vec_size),
            minval=self.lb,
            maxval=self.ub,
        )
        return NSGA2State(
            population=population,
            fitnesses=jnp.zeros((self.pop_size, self.num_objectives)),
            ranks=jnp.zeros((self.pop_size,), dtype=jnp.int32),
            crowding=jnp.zeros((self.pop_size,)),
            offspring=population,
            initialized=jnp.bool_(False),
            key=key,
        )

    def ask(self, state: NSGA2State) -> tuple[Params, NSGA2State]:
        def breed(state):
            key, selection_key, crossover_key, mutation_key = jax.random.split(
                state.key, 4
            )
            num_pairs = (self.pop_size + 1) // 2
            contestants = jax.random.randint(
                selection_key, (2 * num_pairs, 2), 0, self.pop_size
            )
            left, right = contestants[:, 0], contestants[:, 1]
            left_wins = (state.ranks[left] < state.ranks[right]) | (
                (state.ranks[left] == state.ranks[right])
                & (state.crowding[left] >= state.crowding[right])
            )
            parents = state.population[jnp.where(left_wins, left, right)]
            offspring = self._crossover(parents, crossover_key)[: self.pop_size]
            offspring = self._mutate(offspring, mutation_key)
            return state.replace(offspring=offspring, key=key)

        state = jax.lax.cond(state.initialized, breed, lambda state: state, state)
        return self.param_vec_spec.to_tree(state.offspring), state

    def tell(
        self, state: NSGA2State, fitnesses: jax.Array
    ) -> tuple[PyTreeDict, NSGA2State]:
        if fitnesses.shape != (self.pop_size, self.num_objectives):
            raise ValueError("fitnesses must have shape (pop_size, num_objectives)")

        def select_survivors(_):
            population = jnp.concatenate((state.population, state.offspring))
            objectives = jnp.concatenate((state.fitnesses, fitnesses))
            ranks = non_dominated_sort(objectives)
            crowding = crowding_distance(objectives, ranks)
            selected = jnp.lexsort((-crowding, ranks))[: self.pop_size]
            return (
                population[selected],
                objectives[selected],
                ranks[selected],
                crowding[selected],
            )

        def initialize_parents(_):
            ranks = non_dominated_sort(fitnesses)
            return (
                state.offspring,
                fitnesses,
                ranks,
                crowding_distance(fitnesses, ranks),
            )

        population, fitnesses, ranks, crowding = jax.lax.cond(
            state.initialized, select_survivors, initialize_parents, operand=None
        )
        return PyTreeDict(), state.replace(
            population=population,
            fitnesses=fitnesses,
            ranks=ranks,
            crowding=crowding,
            initialized=jnp.bool_(True),
        )

    def _crossover(self, parents: jax.Array, key: jax.Array) -> jax.Array:
        """Apply bounded simulated binary crossover to pairs of parents."""
        sample_key, pair_key, gene_key, swap_key = jax.random.split(key, 4)
        first, second = parents[::2], parents[1::2]
        lower, upper = jnp.minimum(first, second), jnp.maximum(first, second)
        different = upper - lower > 1e-14
        difference = jnp.where(different, upper - lower, 1)
        uniform = jax.random.uniform(sample_key, lower.shape)
        exponent = 1.0 / (self.crossover_eta + 1)

        def spread(beta):
            alpha = 2 - beta ** -(self.crossover_eta + 1)
            return jnp.where(
                uniform <= 1 / alpha,
                (uniform * alpha) ** exponent,
                (1 / (2 - uniform * alpha)) ** exponent,
            )

        left_spread = spread(1 + 2 * (lower - self.lb) / difference)
        right_spread = spread(1 + 2 * (self.ub - upper) / difference)
        child_left = 0.5 * (lower + upper - left_spread * difference)
        child_right = 0.5 * (lower + upper + right_spread * difference)
        swap = jax.random.bernoulli(swap_key, shape=lower.shape)
        child_first = jnp.where(swap, child_right, child_left)
        child_second = jnp.where(swap, child_left, child_right)
        crossover = jax.random.uniform(pair_key, (lower.shape[0], 1))
        crossover = crossover < self.crossover_probability
        crossover &= jax.random.bernoulli(gene_key, shape=lower.shape) & different
        children = jnp.stack(
            (
                jnp.where(crossover, child_first, first),
                jnp.where(crossover, child_second, second),
            ),
            axis=1,
        ).reshape(parents.shape)
        return jnp.clip(children, self.lb, self.ub)

    def _mutate(self, population: jax.Array, key: jax.Array) -> jax.Array:
        """Apply bounded polynomial mutation independently to each coordinate."""
        sample_key, mask_key = jax.random.split(key)
        uniform = jax.random.uniform(sample_key, population.shape)
        width = self.ub - self.lb
        lower = (population - self.lb) / width
        upper = (self.ub - population) / width
        power = self.mutation_eta + 1
        left = 2 * uniform + (1 - 2 * uniform) * (1 - lower) ** power
        right = 2 * (1 - uniform) + 2 * (uniform - 0.5) * (1 - upper) ** power
        delta = jnp.where(
            uniform <= 0.5,
            jnp.maximum(left, 0) ** (1 / power) - 1,
            1 - jnp.maximum(right, 0) ** (1 / power),
        )
        mutate = (
            jax.random.uniform(mask_key, population.shape) < self.mutation_probability
        )
        return jnp.clip(
            population + jnp.where(mutate, delta * width, 0), self.lb, self.ub
        )


def crowding_distance(fitnesses: jax.Array, ranks: jax.Array) -> jax.Array:
    """Compute normalized NSGA-II crowding distances within each Pareto front.

    Constant objectives contribute zero. For each nonconstant objective, the
    two boundary solutions of a front receive infinite distance.

    Args:
        fitnesses: Objective values with shape `(population, objectives)`.
        ranks: Nonnegative Pareto ranks with shape `(population,)`.

    Returns:
        Distances with shape `(population,)`; larger values favor diversity.
    """
    size = fitnesses.shape[0]
    distance = jnp.zeros((size,), dtype=fitnesses.dtype)
    for objective in range(fitnesses.shape[1]):
        values = fitnesses[:, objective]
        order = jnp.lexsort((values, ranks))
        sorted_values, sorted_ranks = values[order], ranks[order]
        lower = jax.ops.segment_min(values, ranks, num_segments=size)
        upper = jax.ops.segment_max(values, ranks, num_segments=size)
        span = (upper - lower)[sorted_ranks]
        previous = jnp.roll(sorted_values, 1)
        following = jnp.roll(sorted_values, -1)
        boundary = (jnp.arange(size) == 0) | (jnp.arange(size) == size - 1)
        boundary |= sorted_ranks != jnp.roll(sorted_ranks, 1)
        boundary |= sorted_ranks != jnp.roll(sorted_ranks, -1)
        contribution = jnp.where(
            span > 0,
            jnp.where(
                boundary,
                jnp.inf,
                (following - previous) / jnp.where(span > 0, span, 1),
            ),
            0,
        )
        distance = distance.at[order].add(contribution)
    return distance
