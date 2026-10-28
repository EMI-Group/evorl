import chex
import jax
import jax.numpy as jnp

from evorl.types import PyTreeNode


def tournament_selection(
    fitnesses: jax.Array,
    num_offspring: int,
    key: jax.Array,
    *,
    tournament_size: int,
):
    """Tournament selection operator for single objective."""
    chex.assert_shape(fitnesses, (fitnesses.shape[0],))
    assert num_offspring > 0, "num_offspring must be positive"
    assert tournament_size > 1, "tournament_size must be greater than 1"

    ranks = jnp.argsort(fitnesses, descending=True)
    pop_size = len(ranks)

    selected_indices = jnp.min(
        jax.random.randint(key, (num_offspring, tournament_size), 0, pop_size), axis=-1
    )
    return ranks[selected_indices]


class TournamentSelection(PyTreeNode):
    tournament_size: int = 2

    def __call__(self, fitnesses, num_offspring, key):
        return tournament_selection(
            fitnesses, num_offspring, key, tournament_size=self.tournament_size
        )
