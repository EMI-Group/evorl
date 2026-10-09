from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp


def shmap_vmap(fn: Callable, in_specs, out_specs, *, vmap_in_axes=0, **kwargs):
    """Apply ``vmap`` within each shard of a global logical batch.

    ``shard_map`` keeps the batch axis, giving each device a block of inputs;
    ``vmap`` maps over the rows of that block. The batch size is independent
    of the number of devices, subject to divisibility by the partition size.
    """

    def shmap_f(*args):
        return jax.vmap(fn, in_axes=vmap_in_axes)(*args)

    return jax.shard_map(shmap_f, in_specs=in_specs, out_specs=out_specs, **kwargs)


def shmap_map(fn: Callable, in_specs, out_specs, **kwargs):
    """Sequential execution on different gpu.

    Args:
        fn: function to be executed, only positional arguments are supported
        in_specs: input partition specs in the active mesh
        out_specs: output partition specs in the active mesh
    Returns:
        A wrapped function.
    """

    def g(carry):
        return fn(*carry)

    def shmap_f(*args):
        return jax.lax.map(g, args)

    return jax.shard_map(shmap_f, in_specs=in_specs, out_specs=out_specs, **kwargs)


def tree_scatter_set(
    src: chex.ArrayTree,
    updates: chex.ArrayTree,
    indices: jax.Array,
    spec: jax.P,
) -> chex.ArrayTree:
    """Replace global population rows while keeping the population sharded.

    Uses a one-dimensional mesh partitioning population axis 0 and unique
    global row indices. Indices and selected updates are replicated. Each device writes only the
    rows it owns; out-of-shard indices are dropped without negative wrapping.
    """
    axis_name = spec[0]

    def update_local(local_src, updates, indices):
        rank = jax.lax.axis_index(axis_name)

        def scatter(x, y):
            local_indices = indices - rank * x.shape[0]
            # Use a positive out-of-bounds sentinel for rows owned elsewhere.
            # This also avoids negative-index scatter issues in JAX 0.7.
            local_indices = jnp.where(
                (local_indices >= 0) & (local_indices < x.shape[0]),
                local_indices,
                x.shape[0],
            )
            return x.at[local_indices].set(y, mode="drop")

        return jax.tree.map(scatter, local_src, updates)

    return jax.shard_map(
        update_local,
        in_specs=(spec, jax.P(), jax.P()),
        out_specs=spec,
    )(src, updates, indices)
