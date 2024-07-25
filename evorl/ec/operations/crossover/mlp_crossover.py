from functools import partial

import chex
import jax
import jax.tree_util as jtu


def mlp_crossover(
    key: chex.PRNGKey,
    x1: chex.ArrayTree,
    x2: chex.ArrayTree,
    *,
    num_crossover_frac: float = 1.0,
):
    chex.assert_trees_all_equal_shapes_and_dtypes(x1, x2)

    leaves1, treedef = jtu.tree_flatten_with_path(x1)
    leaves2 = jtu.tree_leaves(x2)

    params1 = []
    params2 = []
    for i, ((path, param1), param2) in enumerate(zip(leaves1, leaves2)):
        if param1.ndim == 2:  # kernel
            key, ind_key, choice_key = jax.random.split(key, num=3)

            # fixed number of crossover op, this is different from the original ERL,
            # which use np.random.randint(num_variables * 2)
            num_crossover = round(param1.shape[0] * num_crossover_frac)

            key, ind_key, choice_key = jax.random.split(key, num=3)

            ind = jax.random.randint(ind_key, (num_crossover,), 0, param1.shape[0])

            param1, param2 = jax.lax.cond(
                jax.random.uniform(choice_key) < 0.5,
                lambda: (param1.at[ind, :].set(param2[ind, :]), param2),
                lambda: (param1, param2.at[ind, :].set(param1[ind, :])),
            )

        elif param1.ndim == 1:
            key, ind_key, choice_key = jax.random.split(key, num=3)

            num_crossover = round(param1.shape[0] * num_crossover_frac)

            ind = jax.random.randint(ind_key, (num_crossover,), 0, param1.shape[0])

            param1, param2 = jax.lax.cond(
                jax.random.uniform(choice_key) < 0.5,
                lambda: (param1.at[ind].set(param2[ind]), param2),
                lambda: (param1, param2.at[ind].set(param1[ind])),
            )

        else:
            raise ValueError(f"Unsupported parameter shape: {param1.shape}")

        params1.append(param1)
        params2.append(param2)

    return jtu.tree_unflatten(treedef, params1), jtu.tree_unflatten(treedef, params2)


class MLPCrossover:
    def __init__(self, num_crossover_frac: float = 1.0):
        self.num_crossover_frac = num_crossover_frac
        self.crossover_fn = jax.vmap(
            partial(mlp_crossover, num_crossover_frac=num_crossover_frac),
            in_axes=(None, 0, 0),
        )

    def __call__(self, key: chex.PRNGKey, xs: chex.ArrayTree):
        n = jtu.tree_leaves(xs)[0].shape[0]
        n = round(n // 2 * 2)
        xs = jtu.tree_map(lambda p: p[:n], xs)
        parents1 = jtu.tree_map(lambda p: p[::2], xs)
        parents2 = jtu.tree_map(lambda p: p[1::2], xs)

        return self.crossover_fn(key, parents1, parents2)
