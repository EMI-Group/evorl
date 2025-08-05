from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import PyTreeNode
from ..utils import is_layer_norm_layer


def mlp_mutate(
    x: chex.ArrayTree,
    key: chex.PRNGKey,
    *,
    weight_max_magnitude: float = 10,
    mut_strength: float = 0.01,
    vector_num_mutation_frac: float = 0.0,
    matrix_num_mutation_frac: float = 0.01,
):
    """Mutation for MLP.

    Args:
        key: PRNGKey
        x: single individual,
        vec_relative_prob: probability of mutating a vector(1-d) parameter.
            Disable vector mutation when set 0.0; ERL use 0.04
    """
    leaves, treedef = jtu.tree_flatten_with_path(x)

    def _mutate(param, key, num_mutation_frac):
        num_mutations = round(num_mutation_frac * param.size)
        key, ind_key, normal_update_key = jax.random.split(key, 3)
        # unlike ERL, we sample elements without replacement
        flat_ind = jax.random.choice(
            ind_key, param.size, (num_mutations,), replace=False
        )
        ind = jnp.unravel_index(flat_ind, param.shape)
        updates = jax.random.normal(normal_update_key, (num_mutations,)) * mut_strength
        param = param.at[ind].set(param[ind] + updates, unique_indices=True)
        param = jnp.clip(param, -weight_max_magnitude, weight_max_magnitude)
        return param

    params = []
    for i, (path, param) in enumerate(leaves):
        if is_layer_norm_layer(path):
            params.append(param)
            continue

        if param.ndim == 2:  # kernel
            param = _mutate(param, key, matrix_num_mutation_frac)
        elif param.ndim == 1:  # bias or layer norm
            if vector_num_mutation_frac > 0:
                param = _mutate(param, key, vector_num_mutation_frac)
        else:
            raise ValueError(f"Unsupported parameter shape: {param.shape}")

        params.append(param)

    return jtu.tree_unflatten(treedef, params)


class MLPMutation(PyTreeNode):
    weight_max_magnitude: float = 10
    mut_strength: float = 0.01
    vector_num_mutation_frac: float = 0.0
    matrix_num_mutation_frac: float = 0.01

    def __post_init__(self):
        assert 0 <= self.vector_num_mutation_frac <= 1, (
            "vector_num_mutation_frac should be in [0, 1]"
        )
        assert 0 <= self.matrix_num_mutation_frac <= 1, (
            "matrix_num_mutation_frac should be in [0, 1]"
        )

        self.mutate_fn = jax.vmap(
            partial(
                mlp_mutate,
                weight_max_magnitude=self.weight_max_magnitude,
                mut_strength=self.mut_strength,
                vector_num_mutation_frac=self.vector_num_mutation_frac,
                matrix_num_mutation_frac=self.matrix_num_mutation_frac,
            ),
        )

    def __call__(self, xs: chex.ArrayTree, key: chex.PRNGKey):
        pop_size = jtu.tree_leaves(xs)[0].shape[0]
        if key.ndim <= 1:
            key = jax.random.split(key, pop_size)
        else:
            chex.assert_shape(
                key,
                (pop_size, 2),
                custom_message=f"Batched key shape {key.shape} must match pop_size: {pop_size}",
            )
        return self.mutate_fn(xs, key)
