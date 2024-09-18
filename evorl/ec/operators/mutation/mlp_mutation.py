from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ..utils import is_layer_norm_layer


def mlp_mutate(
    x: chex.ArrayTree,
    key: chex.PRNGKey,
    *,
    weight_max_magnitude: float = 1e6,
    mut_strength: float = 0.1,
    num_mutation_frac: float = 0.1,
    super_mut_strength: float = 10.0,
    super_mut_prob: float = 0.05,
    reset_prob: float = 0.1,
    vec_relative_prob: float = 0.0,
):
    """
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
                ind1_key,
                ind2_key,
                prob_key,
                normal_update_key,
                super_update_key,
                reset_update_key,
                ssne_prob_key,
            ) = jax.random.split(key, 8)

            ind_dim1 = jax.random.randint(ind1_key, (num_mutations,), 0, param.shape[0])
            ind_dim2 = jax.random.randint(ind2_key, (num_mutations,), 0, param.shape[1])

            prob = jax.random.uniform(prob_key, (num_mutations,))
            super_mask = prob < super_mut_prob
            reset_mask = jnp.logical_and(prob >= super_mut_prob, prob < reset_prob)

            updates = jax.random.uniform(normal_update_key, (num_mutations,)) * jnp.abs(
                param[ind_dim1, ind_dim2]
            )
            updates = jnp.where(
                super_mask, updates * super_mut_strength, updates * mut_strength
            )

            reset_param = jax.random.normal(reset_update_key, (num_mutations,))
            new_param = param.at[ind_dim1, ind_dim2].set(
                jnp.where(reset_mask, reset_param, param[ind_dim1, ind_dim2] + updates)
            )

            ssne_prob = jax.random.uniform(ssne_prob_key)
            param = jnp.where(ssne_prob < ssne_probs[i], new_param, param)

            param = jnp.clip(param, -weight_max_magnitude, weight_max_magnitude)

        elif param.ndim == 1:  # bias or layer norm
            if vec_relative_prob > 0:
                num_mutations = round(num_mutation_frac * param.size)

                (
                    key,
                    ind1_key,
                    prob_key,
                    normal_update_key,
                    super_update_key,
                    reset_update_key,
                    ssne_prob_key,
                ) = jax.random.split(key, 7)

                ind_dim1 = jax.random.randint(
                    ind1_key, (num_mutations,), 0, param.shape[0]
                )

                prob = jax.random.uniform(prob_key, (num_mutations,))
                super_mask = prob < super_mut_prob
                reset_mask = jnp.logical_and(prob >= super_mut_prob, prob < reset_prob)

                updates = jax.random.uniform(
                    normal_update_key, (num_mutations,)
                ) * jnp.abs(param[ind_dim1, ind_dim2])
                updates = jnp.where(
                    super_mask, updates * super_mut_strength, updates * mut_strength
                )

                reset_param = jax.random.normal(reset_update_key, (num_mutations,))
                new_param = param.at[ind_dim1, ind_dim2].set(
                    jnp.where(
                        reset_mask, reset_param, param[ind_dim1, ind_dim2] + updates
                    )
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


class MLPMutation:
    def __init__(
        self,
        weight_max_magnitude: float = 1e6,
        mut_strength: float = 0.1,
        num_mutation_frac: float = 0.1,
        super_mut_strength: float = 10.0,
        super_mut_prob: float = 0.05,
        reset_prob: float = 0.1,
        vec_relative_prob: float = 0.0,
    ):
        self.weight_max_magnitude = weight_max_magnitude
        self.mut_strength = mut_strength
        self.num_mutation_frac = num_mutation_frac
        self.super_mut_strength = super_mut_strength
        self.super_mut_prob = super_mut_prob
        self.reset_prob = reset_prob
        self.vec_relative_prob = vec_relative_prob

        self.mutate_fn = jax.vmap(
            partial(
                mlp_mutate,
                weight_max_magnitude=weight_max_magnitude,
                mut_strength=mut_strength,
                num_mutation_frac=num_mutation_frac,
                super_mut_strength=super_mut_strength,
                super_mut_prob=super_mut_prob,
                reset_prob=reset_prob,
                vec_relative_prob=vec_relative_prob,
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
