import chex
import jax
import jax.numpy as jnp

from evorl.types import PyTreeNode


def tournament_selection(
    fitnesses: chex.Array,
    num_offsprings: int,
    key: chex.PRNGKey,
    *,
    tournament_size: int,
):
    """Tournament selection operator for single objective."""
    chex.assert_shape(fitnesses, (fitnesses.shape[0],))
    assert num_offsprings > 0, "num_offsprings must be positive"
    assert tournament_size > 1, "tournament_size must be greater than 1"

    ranks = jnp.argsort(fitnesses, descending=True)
    pop_size = len(ranks)

    selected_indices = jnp.min(
        jax.random.randint(key, (num_offsprings, tournament_size), 0, pop_size), axis=-1
    )
    return ranks[selected_indices]


class TournamentSelection(PyTreeNode):
    tournament_size: int = 2

    def __call__(self, fitnesses, num_offsprings, key):
        return tournament_selection(
            fitnesses, num_offsprings, key, tournament_size=self.tournament_size
        )
