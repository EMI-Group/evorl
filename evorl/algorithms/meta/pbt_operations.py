import chex
import jax
import jax.numpy as jnp
from evorl.types import PyTreeDict


def explore(
    parent: chex.ArrayTree,
    key: chex.PRNGKey,
    perturb_factor: dict[str, float],
    search_space: dict[str, dict[str, float]],
):
    """Define the exploration operation for PBT.

    Normally explore the local of an individual.
    i.e., mutation op in the context of EC.
    Here we use the orginal exploration operator in PBT.
    """
    offspring = PyTreeDict()
    for hp_name in parent.keys():
        val = parent[hp_name] * (
            1
            + jax.random.uniform(
                key,
                minval=-perturb_factor[hp_name],
                maxval=perturb_factor[hp_name],
            )
        )
        offspring[hp_name] = jnp.clip(
            val, min=search_space[hp_name]["low"], max=search_space[hp_name]["high"]
        )

    return offspring


def select(
    pop_episode_returns: chex.Array, key: chex.PRNGKey, bottoms_num: int, tops_num: int
):
    """Select parents to replace worse individuals."""
    indices = jnp.argsort(pop_episode_returns)
    bottoms_indices = indices[:bottoms_num]
    tops_indices = indices[-tops_num:]

    # replace bottoms with random tops
    tops_choice_indices = jax.random.choice(
        key, tops_indices, (bottoms_num,), replace=True
    )  # ensure selecting (pop_size*bottom_ratio) parents from top

    return tops_choice_indices, bottoms_indices
