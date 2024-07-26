from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def mlp_mutate(
    key: chex.PRNGKey,
    x: chex.ArrayTree,
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
    ssne_probs = jax.random.uniform(ssne_key, (len(leaves),)) * 2

    params = []
    for i, (path, param) in enumerate(leaves):
        if param.ndim == 2:  # kernel
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
            # normal_mask = jnp.logical_not(jnp.logical_or(super_mask, reset_mask))

            updates = (
                jax.random.uniform(normal_update_key, (num_mutations,))
                * mut_strength
                * jnp.abs(param[ind_dim1, ind_dim2])
            )
            super_updates = (
                jax.random.normal(super_update_key, (num_mutations,))
                * super_mut_strength
                * jnp.abs(param[ind_dim1, ind_dim2])
            )
            reset_updates = jax.random.normal(reset_update_key, (num_mutations,))

            updates = jnp.where(super_mask, super_updates, updates)
            updates = jnp.where(reset_mask, reset_updates, updates)

            updates = jnp.clip(updates, -weight_max_magnitude, weight_max_magnitude)

            ssne_prob = jax.random.uniform(ssne_prob_key)
            updates = jnp.where(
                ssne_prob < ssne_probs[i], updates, jnp.zeros_like(updates)
            )

            param = param.at[ind_dim1, ind_dim2].add(updates)

        elif param.ndim == 1:  # bias or layer norm
            if vec_relative_prob > 0:
                # Note: We use fixed number of mutations for 1-d params,
                # This is a little different from the original ERL
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

                updates = (
                    jax.random.uniform(normal_update_key, (num_mutations,))
                    * mut_strength
                    * jnp.abs(param[ind_dim1])
                )
                super_updates = (
                    jax.random.normal(super_update_key, (num_mutations,))
                    * super_mut_strength
                    * jnp.abs(param[ind_dim1])
                )
                reset_updates = jax.random.normal(reset_update_key, (num_mutations,))

                updates = jnp.where(super_mask, super_updates, updates)
                updates = jnp.where(reset_mask, reset_updates, updates)

                updates = jnp.clip(updates, -weight_max_magnitude, weight_max_magnitude)

                ssne_prob = jax.random.uniform(ssne_prob_key)
                updates = jnp.where(
                    ssne_prob < ssne_probs[i] * vec_relative_prob,
                    updates,
                    jnp.zeros_like(updates),
                )

                param = param.at[ind_dim1].add(updates)

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
            in_axes=(None, 0),
        )

    def __call__(self, key: chex.PRNGKey, xs: chex.ArrayTree):
        return self.mutate_fn(key, xs)
