from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp


def mark_varying(x, axis_name: str | tuple[str, ...] | None):
    """Allow pytree leaves to vary across explicitly supplied Manual axes.

    Use this before a local loop or branch when its initial values or constant
    outputs must have the same VMA types as values computed from local data.
    Leaves already varying over the supplied axes are returned unchanged.

    Args:
        x: Pytree of arrays to mark.
        axis_name: Required axis name or tuple of names active as Manual axes
            in the enclosing `shard_map`. Pass `None` or an empty tuple to
            return `x` unchanged. Callers may obtain the current axes from
            `jax.sharding.get_abstract_mesh().manual_axes`; this function
            does not query the mesh itself.

    Returns:
        Pytree with the same values and shapes, marked varying over the named
        axes. The casts perform no forward communication; their reverse-mode
        transpose can introduce a sum reduction.
    """
    if axis_name is None:
        return x
    axes = (axis_name,) if isinstance(axis_name, str) else axis_name
    if not axes:
        return x

    def cast(leaf):
        varying = jax.typeof(leaf).manual_axis_type.varying
        missing = tuple(axis for axis in axes if axis not in varying)
        if not missing:
            return leaf
        return jax.lax.pcast(leaf, missing, to="varying")

    return jax.tree.map(cast, x)


def shmap_vmap(fn: Callable, in_specs, out_specs, *, vmap_in_axes=0, **kwargs):
    """Apply `vmap` within each shard of a global logical batch.

    `shard_map` keeps the batch axis, giving each device a block of inputs;
    `vmap` maps over the rows of that block. The batch size is independent
    of the number of devices, subject to divisibility by the partition size.

    Args:
        fn: Function to apply to each row of the local batch.
        in_specs: Input partition specs in the active mesh.
        out_specs: Output partition specs in the active mesh.
        vmap_in_axes: Input axes forwarded to `jax.vmap`.
        **kwargs: Additional keyword arguments forwarded to `jax.shard_map`.

    Returns:
        A callable that applies `vmap(fn)` within each shard.
    """

    def shmap_f(*args):
        return jax.vmap(fn, in_axes=vmap_in_axes)(*args)

    return jax.shard_map(shmap_f, in_specs=in_specs, out_specs=out_specs, **kwargs)


def shmap_map(fn: Callable, in_specs, out_specs, **kwargs):
    """Map `fn` sequentially over each device's local batch.

    Args:
        fn: Function accepting positional arguments for one batch element.
        in_specs: Input partition specs in the active mesh.
        out_specs: Output partition specs in the active mesh.
        **kwargs: Additional keyword arguments forwarded to `jax.shard_map`.

    Returns:
        A callable that applies `jax.lax.map` within each shard.
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
    global row indices. Indices and selected updates are replicated. Each
    device writes only the rows it owns; out-of-shard indices are dropped
    without negative wrapping.

    Args:
        src: Pytree whose leading population dimension is sharded.
        updates: Replicated replacement rows with the same pytree structure.
        indices: Replicated, unique global indices of the rows to replace.
        spec: Partition spec naming the population axis in dimension zero.

    Returns:
        A pytree containing the updated population, with the original sharding.
    """
    axis_name = spec[0]

    def update_local(local_src, updates, indices):
        rank = jax.lax.axis_index(axis_name)

        def scatter(x, y):
            local_indices = indices - rank * x.shape[0]
            # Use a positive out-of-bounds sentinel for rows owned elsewhere.
            # This also avoids negative-index scatter issues.
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
