import math

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import PyTreeData, PyTreeDict
from evorl.ec.operators import ERLMutation, MLPCrossover, TournamentSelection
from evorl.utils.jax_utils import tree_get

from .ec_optimizer import EvoOptimizer


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
    num_crossover_frac: float = 0.1

    def __post_init__(self):
        assert self.pop_size >= self.num_elites, "num_elites must be <= pop_size"
        assert (
            self.pop_size - self.num_elites
        ) % 2 == 0 or not self.enable_crossover, (
            "(pop_size - num_elites) must be even when enable crossover"
        )
        self.select_parents = TournamentSelection(tournament_size=self.tournament_size)
        self.mutate = ERLMutation(
            weight_max_magnitude=self.weight_max_magnitude,
            mut_strength=self.mut_strength,
            num_mutation_frac=self.num_mutation_frac,
            super_mut_strength=self.super_mut_strength,
            super_mut_prob=self.super_mut_prob,
            reset_prob=self.reset_prob,
            vec_relative_prob=self.vec_relative_prob,
        )
        if self.enable_crossover:
            self.crossover = MLPCrossover(num_crossover_frac=self.num_crossover_frac)

    def init(self, pop, key) -> ERLGAState:
        return ERLGAState(pop=pop, key=key)

    def ask(self, state: ERLGAState) -> tuple[chex.ArrayTree, ERLGAState]:
        return state.pop, state

    def tell(
        self, state: ERLGAState, fitnesses: chex.Array
    ) -> tuple[PyTreeDict, ERLGAState]:
        # Note: We simplify the update in ERL
        key, select_key, mutate_key, crossover_key = jax.random.split(state.key, 4)

        elite_indices = jnp.argsort(fitnesses, descending=True)[: self.num_elites]
        elites = tree_get(state.pop, elite_indices)

        if self.enable_crossover:
            real_num_parents = self.pop_size - self.num_elites
            num_parents = math.ceil((real_num_parents) / 2) * 2
            parents_indices = self.select_parents(fitnesses, num_parents, select_key)
            parents = tree_get(state.pop, parents_indices)

            offsprings = self.crossover(parents, crossover_key)
            if real_num_parents % 2 != 0:
                offsprings = tree_get(offsprings, slice(real_num_parents))
            offsprings = self.mutate(offsprings, mutate_key)
        else:
            num_parents = self.pop_size - self.num_elites
            parents_indices = self.select_parents(fitnesses, num_parents, select_key)
            parents = tree_get(state.pop, parents_indices)
            offsprings = self.mutate(parents, mutate_key)

        new_pop = jtu.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0), elites, offsprings
        )

        return PyTreeDict(), state.replace(pop=new_pop, key=key)


class ERLGAModState(ERLGAState):
    external_pop: None | chex.ArrayTree = None


class ERLGAMod(ERLGA):
    external_size: int

    def __post_init__(self):
        assert self.pop_size >= (self.num_elites + self.external_size), (
            "num_elites+external_size must be <= pop_size"
        )
        super().__post_init__()

    def init(self, pop, key) -> ERLGAModState:
        return ERLGAModState(pop=pop, key=key)

    def tell_external(
        self, state: ERLGAModState, fitnesses: chex.Array
    ) -> tuple[PyTreeDict, ERLGAModState]:
        # Note: We simplify the update in ERL
        key, select_key, mutate_key, crossover_key = jax.random.split(state.key, 4)

        sorted_indices = jnp.argsort(fitnesses, descending=True)
        elite_indices = sorted_indices[: self.num_elites]
        elites = tree_get(state.pop, elite_indices)

        # unselected(worst) are replaced by external op (e.g: from RL)
        # unselected_indices = sorted_indices[-self.external_size :]
        unselected = state.external_pop

        selected_indices = sorted_indices[: -self.external_size]

        if self.enable_crossover:
            real_num_parents = self.pop_size - self.num_elites - self.external_size
            num_parents = math.ceil((real_num_parents) / 2) * 2
            parents_indices = selected_indices[
                self.select_parents(
                    fitnesses[selected_indices],
                    num_parents,
                    select_key,
                )
            ]
            parents = tree_get(state.pop, parents_indices)
            offsprings = self.crossover(parents, crossover_key)
            if real_num_parents % 2 != 0:
                offsprings = tree_get(offsprings, slice(real_num_parents))
            offsprings = self.mutate(offsprings, mutate_key)
        else:
            num_parents = self.pop_size - self.num_elites - self.external_size
            parents_indices = selected_indices[
                self.select_parents(
                    fitnesses[selected_indices],
                    num_parents,
                    select_key,
                )
            ]
            parents = tree_get(state.pop, parents_indices)
            offsprings = self.mutate(parents, mutate_key)

        new_pop = jtu.tree_map(
            lambda x, y, z: jnp.concatenate([x, y, z], axis=0),
            elites,
            offsprings,
            unselected,
        )

        return PyTreeDict(), state.replace(pop=new_pop, key=key, external_pop=None)
