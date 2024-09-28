from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import PyTreeData, pytree_field
from evorl.ec.operators import MLPMutation, MLPCrossover, TournamentSelection

from .ec_optimizer import EvoOptimizer, ECState


class ERLGAState(PyTreeData):
    pop: chex.ArrayTree
    key: chex.PRNGKey


class ERLGA(EvoOptimizer):
    pop_size: int
    num_elites: int

    # selection
    tournament_size: int = 3

    # mutation
    weight_max_magnitude: float = 1e6
    mut_strength: float = 0.1
    num_mutation_frac: float = 0.1
    super_mut_strength: float = 10.0
    super_mut_prob: float = 0.05
    reset_prob: float = 0.1
    vec_relative_prob: float = 0.0

    # crossover
    enable_crossover: bool = True
    num_crossover_frac: float = 2

    # op
    select_parents: Callable = pytree_field(lazy_init=True)
    mutate: Callable = pytree_field(lazy_init=True)
    crossover: Callable = pytree_field(lazy_init=True)

    def __post_init__(self):
        assert (
            self.pop_size - self.num_elites
        ) % 2 == 0, "pop_size - num_elites must be even"

        selection_op = TournamentSelection(tournament_size=self.tournament_size)
        mutation_op = MLPMutation(
            weight_max_magnitude=self.weight_max_magnitude,
            mut_strength=self.mut_strength,
            num_mutation_frac=self.num_mutation_frac,
            super_mut_strength=self.super_mut_strength,
            super_mut_prob=self.super_mut_prob,
            reset_prob=self.reset_prob,
            vec_relative_prob=self.vec_relative_prob,
        )
        crossover_op = MLPCrossover(num_crossover_frac=self.num_crossover_frac)

        self.set_frozen_attr("select_parents", selection_op)
        self.set_frozen_attr("mutate", mutation_op)
        self.set_frozen_attr("crossover", crossover_op)

    def init(self, pop, key) -> ECState:
        return ERLGAState(pop=pop, key=key)

    def tell(self, state: ECState, xs: chex.ArrayTree, fitnesses: chex.Array):
        # Note: We simplify the update in ERL
        key, select_key, mutate_key, crossover_key = jax.random.split(state.key, 4)

        elite_indices = jnp.argsort(fitnesses, descending=True)[: self.num_elites]
        elites = jtu.tree_map(lambda x: x[elite_indices], state.pop)

        parents_indices = self.select_parents(
            fitnesses, self.pop_size - self.num_elites, select_key
        )
        parents = jtu.tree_map(lambda x: x[parents_indices], state.pop)

        if self.enable_crossover:
            offsprings = self.crossover(parents, crossover_key)
            offsprings = self.mutate(offsprings, mutate_key)
        else:
            offsprings = self.mutate(parents, mutate_key)

        new_pop = jtu.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0), elites, offsprings
        )

        return ERLGAState(pop=new_pop, key=key)

    def ask(self, state, key):
        return state.pop
