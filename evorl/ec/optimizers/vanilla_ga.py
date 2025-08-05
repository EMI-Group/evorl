import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import PyTreeData, PyTreeDict
from evorl.ec.operators import MLPMutation, MLPCrossover, TournamentSelection

from .ec_optimizer import EvoOptimizer


class VanillaGAState(PyTreeData):
    """State of the VanillaGA."""

    pop: chex.ArrayTree
    key: chex.PRNGKey


class VanillaGA(EvoOptimizer):
    """Vanilla Genetic Algorithm.

    The Genetic Algorithm used in the original ERL.
    Paper: [Evolution-Guided Policy Gradient in Reinforcement Learning](https://arxiv.org/abs/1805.07917)
    """

    pop_size: int
    num_elites: int

    # selection
    tournament_size: int = 2

    # mutation
    weight_max_magnitude: float = 1e6
    mut_strength: float = 0.1
    vector_num_mutation_frac: float = 0.0
    matrix_num_mutation_frac: float = 0.01

    # crossover
    enable_crossover: bool = True
    num_crossover_frac: float = 0.1

    def __post_init__(self):
        assert (
            self.pop_size - self.num_elites
        ) % 2 == 0 or not self.enable_crossover, (
            "(pop_size - num_elites) must be even when enable crossover"
        )

        self.select_parents = TournamentSelection(tournament_size=self.tournament_size)
        self.mutate = MLPMutation(
            weight_max_magnitude=self.weight_max_magnitude,
            mut_strength=self.mut_strength,
            vector_num_mutation_frac=self.vector_num_mutation_frac,
            matrix_num_mutation_frac=self.matrix_num_mutation_frac,
        )
        if self.enable_crossover:
            self.crossover = MLPCrossover(num_crossover_frac=self.num_crossover_frac)

    def init(self, pop: chex.ArrayTree, key: chex.PRNGKey) -> VanillaGAState:
        return VanillaGAState(pop=pop, key=key)

    def ask(self, state: VanillaGAState) -> tuple[chex.ArrayTree, VanillaGAState]:
        return state.pop, state

    def tell(
        self, state: VanillaGAState, fitnesses: chex.Array
    ) -> tuple[PyTreeDict, VanillaGAState]:
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

        return PyTreeDict(), state.replace(pop=new_pop, key=key)
