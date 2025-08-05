from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import PyTreeNode
from ..utils import is_layer_norm_layer


def erl_mutate(
    x: chex.ArrayTree,
    key: chex.PRNGKey,
    *,
    weight_max_magnitude: float = 1e6,
    mut_strength: float = 0.1,
    num_mutation_frac: float = 0.1,
    super_mut_strength: float = 10.0,
    super_mut_prob: float = 0.05,
    reset_prob: float = 0.05,
    vec_relative_prob: float = 0.0,
):
    """Mutation used in the original ERL for MLP.

    Args:
        key: PRNGKey
        x: single individual,
        vec_relative_prob: probability of mutating a vector(1-d) parameter.
            Disable vector mutation when set 0.0; ERL use 0.04
    """
    leaves, treedef = jtu.tree_flatten_with_path(x)
    key, ssne_key = jax.random.split(key)

    # prob thresould of whether mutate a param
    ssne_probs = jax.random.uniform(ssne_key, (len(leaves),)) * 2

    params = []
    for i, (path, param) in enumerate(leaves):
        if is_layer_norm_layer(path):
            params.append(param)
            continue

        if param.ndim == 2:  # kernel
            # Note: We use fixed number of mutations for a param,
            # This is a little different from the original ERL
            num_mutations = round(num_mutation_frac * param.size)

            (
                key,
                ind_key,
                prob_key,
                normal_update_key,
                reset_update_key,
                ssne_prob_key,
            ) = jax.random.split(key, 6)

            # unlike ERL, we sample elements without replacement
            num_param = param.shape[0] * param.shape[1]
            flat_ind = jax.random.choice(
                ind_key, num_param, (num_mutations,), replace=False
            )
            ind = jnp.unravel_index(flat_ind, param.shape)

            prob = jax.random.uniform(prob_key, (num_mutations,))
            super_mask = prob < super_mut_prob
            reset_mask = jnp.logical_and(
                prob >= super_mut_prob, prob < reset_prob + super_mut_prob
            )

            updates = jax.random.normal(normal_update_key, (num_mutations,)) * jnp.abs(
                param[ind]
            )
            updates = jnp.where(
                super_mask, updates * super_mut_strength, updates * mut_strength
            )

            reset_param = jax.random.normal(reset_update_key, (num_mutations,))
            new_param = param.at[ind].set(
                jnp.where(reset_mask, reset_param, param[ind] + updates),
                unique_indices=True,
            )

            ssne_prob = jax.random.uniform(ssne_prob_key)
            param = jnp.where(ssne_prob < ssne_probs[i], new_param, param)

            param = jnp.clip(param, -weight_max_magnitude, weight_max_magnitude)

        elif param.ndim == 1:  # bias or layer norm
            if vec_relative_prob > 0:
                num_mutations = round(num_mutation_frac * param.size)

                (
                    key,
                    ind_key,
                    prob_key,
                    normal_update_key,
                    reset_update_key,
                    ssne_prob_key,
                ) = jax.random.split(key, 6)

                ind = jax.random.choice(ind_key, param.shape[0], (num_mutations,))

                prob = jax.random.uniform(prob_key, (num_mutations,))
                super_mask = prob < super_mut_prob
                reset_mask = jnp.logical_and(prob >= super_mut_prob, prob < reset_prob)

                updates = jax.random.normal(
                    normal_update_key, (num_mutations,)
                ) * jnp.abs(param[ind])
                updates = jnp.where(
                    super_mask, updates * super_mut_strength, updates * mut_strength
                )

                reset_param = jax.random.normal(reset_update_key, (num_mutations,))
                new_param = param.at[ind].set(
                    jnp.where(reset_mask, reset_param, param[ind] + updates),
                    unique_indices=True,
                )

                ssne_prob = jax.random.uniform(ssne_prob_key)
                param = jnp.where(
                    ssne_prob < ssne_probs[i] * vec_relative_prob, new_param, param
                )

                param = jnp.clip(param, -weight_max_magnitude, weight_max_magnitude)

        else:
            raise ValueError(f"Unsupported parameter shape: {param.shape}")

        params.append(param)

    return jtu.tree_unflatten(treedef, params)


class ERLMutation(PyTreeNode):
    weight_max_magnitude: float = 1e6
    mut_strength: float = 0.1
    num_mutation_frac: float = 0.1
    super_mut_strength: float = 10.0
    super_mut_prob: float = 0.05
    reset_prob: float = 0.05
    vec_relative_prob: float = 0.0

    def __post_init__(self):
        assert self.num_mutation_frac >= 0 and self.num_mutation_frac <= 1, (
            "num_mutation_frac should be in [0, 1]"
        )

        self.mutate_fn = jax.vmap(
            partial(
                erl_mutate,
                weight_max_magnitude=self.weight_max_magnitude,
                mut_strength=self.mut_strength,
                num_mutation_frac=self.num_mutation_frac,
                super_mut_strength=self.super_mut_strength,
                super_mut_prob=self.super_mut_prob,
                reset_prob=self.reset_prob,
                vec_relative_prob=self.vec_relative_prob,
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
