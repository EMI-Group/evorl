from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import PyTreeNode
from ..utils import is_layer_norm_layer


def mlp_crossover(
    x1: chex.ArrayTree,
    x2: chex.ArrayTree,
    key: chex.PRNGKey,
    *,
    num_crossover_frac: float = 0.1,
):
    chex.assert_trees_all_equal_shapes_and_dtypes(x1, x2)

    leaves1, treedef = jtu.tree_flatten_with_path(x1)
    leaves2 = jtu.tree_leaves(x2)

    params1 = []
    params2 = []
    for i, ((path, param1), param2) in enumerate(zip(leaves1, leaves2)):
        if is_layer_norm_layer(path):
            # skip layer norm layers
            params1.append(param1)
            params2.append(param2)
            continue

        if param1.ndim <= 2:  # kernel
            # for 2d array, we exchange the rows
            # for 1d array, we exchange the elements
            key, ind_key, choice_key = jax.random.split(key, num=3)

            # we use fixed number of crossover op: sampled without replacement
            # this is different from the original ERL: np.random.randint(num_variables * 2)
            num_crossover = round(param1.shape[0] * num_crossover_frac)

            ind = jax.random.choice(
                ind_key, param1.shape[0], (num_crossover,), replace=False
            )

            mask = jax.random.uniform(choice_key, (num_crossover,)) < 0.5
            if param1.ndim > 1:
                mask = mask[..., None]

            zero_update = jnp.zeros((num_crossover, *param1.shape[1:]))

            param1 = param1.at[ind].set(
                jnp.where(mask, zero_update, param2[ind]),
                unique_indices=True,
            )

            param2 = param2.at[ind].set(
                jnp.where(jnp.logical_not(mask), zero_update, param1[ind]),
                unique_indices=True,
            )

        else:
            raise ValueError(f"Unsupported parameter shape: {param1.shape}")

        params1.append(param1)
        params2.append(param2)

    return jtu.tree_unflatten(treedef, params1), jtu.tree_unflatten(treedef, params2)


class MLPCrossover(PyTreeNode):
    num_crossover_frac: float = 0.1

    def __post_init__(self):
        assert self.num_crossover_frac >= 0 and self.num_crossover_frac <= 1, (
            "num_crossover_frac should be in [0, 1]"
        )

        self.crossover_fn = jax.vmap(
            partial(mlp_crossover, num_crossover_frac=self.num_crossover_frac),
        )

    def __call__(self, xs: chex.ArrayTree, key: chex.PRNGKey):
        pop_size = jtu.tree_leaves(xs)[0].shape[0]
        assert pop_size % 2 == 0, "pop_size must be even"
        # xs = jtu.tree_map(lambda p: p[:n], xs)
        parents1 = jtu.tree_map(lambda x: x[0::2], xs)
        parents2 = jtu.tree_map(lambda x: x[1::2], xs)

        if key.ndim <= 1:
            key = jax.random.split(key, pop_size // 2)
        else:
            chex.assert_shape(
                key,
                (pop_size, 2),
                custom_message=f"Batched key shape {key.shape} must match pop_size: {pop_size}",
            )

        offsprings1, offsprings2 = self.crossover_fn(parents1, parents2, key)
        return jtu.tree_map(
            lambda x1, x2: jnp.concatenate([x1, x2], axis=0), offsprings1, offsprings2
        )
