import os
from collections.abc import Iterable, Sequence, Callable
from functools import partial
import math
import copy

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def disable_gpu_preallocation():
    """Disable GPU memory preallocation for XLA.

    Call this method at the beginning of your script.
    """
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def optimize_gpu_utilization():
    """Possible Optimizations for Nvidia GPU.

    This function is not tested.
    """
    # xla_flags = os.getenv("XLA_FLAGS", "")
    # print(f"current XLA_FLAGS: {xla_flags}")
    # if len(xla_flags) > 0:
    #     xla_flags = xla_flags + " "
    # os.environ['XLA_FLAGS'] = xla_flags + (
    #     '--xla_gpu_enable_triton_softmax_fusion=true '
    #     '--xla_gpu_triton_gemm_any=True '
    #     # '--xla_gpu_enable_async_collectives=true '
    #     # '--xla_gpu_enable_latency_hiding_scheduler=true '
    #     # '--xla_gpu_enable_highest_priority_async_stream=true '
    # )

    # used for single-host multi-device computations on Nvidia GPUs
    os.environ.update(
        {
            "NCCL_LL128_BUFFSIZE": "-2",
            "NCCL_LL_BUFFSIZE": "-2",
            "NCCL_PROTO": "SIMPLE,LL,LL128",
        }
    )


def enable_deterministic_mode():
    """Enable deterministic mode for JAX.

    Call this method at the beginning of your script.
    """
    xla_flags = os.getenv("XLA_FLAGS", "")
    # print(f"current XLA_FLAGS: {xla_flags}")
    if len(xla_flags) > 0:
        xla_flags = xla_flags + " "
    os.environ["XLA_FLAGS"] = xla_flags + "--xla_gpu_deterministic_ops=true"


# use chex.set_n_cpu_devices(n) instead
# def set_host_device_count(n):
#     """
#     By default, XLA considers all CPU cores as one device. This utility tells XLA
#     that there are `n` host (CPU) devices available to use. As a consequence, this
#     allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.

#     .. note:: This utility only takes effect at the beginning of your program.
#         Under the hood, this sets the environment variable
#         `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
#         `[num_device]` is the desired number of CPU devices `n`.

#     .. warning:: Our understanding of the side effects of using the
#         `xla_force_host_platform_device_count` flag in XLA is incomplete. If you
#         observe some strange phenomenon when using this utility, please let us
#         know through our issue or forum page. More information is available in this
#         `JAX issue <https://github.com/google/jax/issues/1408>`_.

#     :param int n: number of devices to use.
#     """
#     xla_flags = os.getenv("XLA_FLAGS", "")
#     xla_flags = re.sub(
#         r"--xla_force_host_platform_device_count=\S+", "", xla_flags).split()
#     os.environ["XLA_FLAGS"] = " ".join(
#         ["--xla_force_host_platform_device_count={}".format(n)] + xla_flags)


def tree_zeros_like(nest: chex.ArrayTree, dtype=None) -> chex.ArrayTree:
    """Pytree version of `jnp.zeros_like`."""
    return jtu.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), nest)


def tree_ones_like(nest: chex.ArrayTree, dtype=None) -> chex.ArrayTree:
    """Pytree version of `jnp.ones_like`."""
    return jtu.tree_map(lambda x: jnp.ones(x.shape, dtype or x.dtype), nest)


def tree_concat(nest1: chex.ArrayTree, nest2: chex.ArrayTree, axis: int = 0):
    """Pytree version of `jnp.concatenate`."""
    return jtu.tree_map(lambda x, y: jnp.concatenate([x, y], axis=axis), nest1, nest2)


def tree_stop_gradient(nest: chex.ArrayTree) -> chex.ArrayTree:
    """Pytree version of `jax.lax.stop_gradient`."""
    return jtu.tree_map(jax.lax.stop_gradient, nest)


def tree_astype(tree: chex.ArrayTree, dtype):
    """Pytree version of `jnp.astype`."""
    return jtu.tree_map(lambda x: x.astype(dtype), tree)


def tree_last(tree: chex.ArrayTree):
    """Get the last element of each array in the pytree."""
    return jtu.tree_map(lambda x: x[-1], tree)


def tree_get(tree: chex.ArrayTree, idx_or_slice):
    """Get the elements of each array in the pytree."""
    return jtu.tree_map(lambda x: x[idx_or_slice], tree)


def tree_set(
    src: chex.ArrayTree,
    target: chex.ArrayTree,
    idx_or_slice,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | None = None,
):
    """Set part of each array in the pytree.

    A Pytree version of `src[idx_or_slice]=target`.

    Args:
        src: The source pytree.
        target: The target pytree.
        idx_or_slice: The indices or slices to be set.
        indices_are_sorted: Whether the indices are sorted.
        unique_indices: Whether the indices are unique.
        mode: The mode to set the values.

    Returns:
        The updated source pytree.
    """
    return jtu.tree_map(
        lambda x, y: x.at[idx_or_slice].set(
            y,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
        ),
        src,
        target,
    )


def scan_and_mean(*args, **kwargs):
    """Scan with mean aggregation.

    Usage: same like `jax.lax.scan`, but the scan results will be averaged.
    """
    last_carry, ys = jax.lax.scan(*args, **kwargs)
    return last_carry, jtu.tree_map(lambda x: x.mean(axis=0), ys)


def scan_and_last(*args, **kwargs):
    """Scan and return last iteration results.

    Usage: same like `jax.lax.scan`, but return the last scan iteration results.
    """
    last_carry, ys = jax.lax.scan(*args, **kwargs)
    return last_carry, jtu.tree_map(lambda x: x[-1] if x.shape[0] > 0 else x, ys)


def jit_method(
    *,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    **kwargs,
):
    """A decorator for `jax.jit` with arguments.

    Args:
        static_argnums: The positional argument indices that are constant across
            different calls to the function.

    Returns:
        A decorator for `jax.jit` with arguments.
    """
    return partial(
        jax.jit,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        **kwargs,
    )


def pmap_method(
    axis_name,
    *,
    static_broadcasted_argnums=(),
    donate_argnums=(),
    **kwargs,
):
    """A decorator for `jax.pmap` with arguments."""
    return partial(
        jax.pmap,
        axis_name,
        static_broadcasted_argnums=static_broadcasted_argnums,
        donate_argnums=donate_argnums,
        **kwargs,
    )


def _vmap_rng_split(key: chex.PRNGKey, num: int = 2) -> chex.PRNGKey:
    """Enhanced version of `jax.random.split` that allows batched keys.

    Args:
        key: Key or batched keys with shape (B, 2)
        num: Number of keys to split.

    Returns:
        Batched keys with shape (num, B, 2)
    """
    chex.assert_shape(key, (..., 2))

    rng_split_fn = jax.random.split

    for _ in range(key.ndim - 1):
        rng_split_fn = jax.vmap(rng_split_fn, in_axes=(0, None), out_axes=1)

    return rng_split_fn(key, num)


def rng_split(key: chex.PRNGKey, num: int = 2) -> chex.PRNGKey:
    """Unified Version of `jax.random.split` for both single key and batched keys."""
    if key.ndim == 1:
        chex.assert_shape(key, (2,))
        return jax.random.split(key, num)
    else:
        return _vmap_rng_split(key, num)


def rng_split_by_shape(key: chex.PRNGKey, shape: tuple[int]) -> chex.PRNGKey:
    """Split the key into multiple keys according to the shape."""
    chex.assert_shape(key, (2,))
    keys = jax.random.split(key, math.prod(shape))
    return jnp.reshape(keys, shape + (2,))


def rng_split_like_tree(
    key: chex.PRNGKey, target: chex.ArrayTree, is_leaf=None
) -> chex.ArrayTree:
    """Split the key according to the structure of the target pytree."""
    treedef = jtu.tree_structure(target, is_leaf=is_leaf)
    keys = jax.random.split(key, treedef.num_leaves)
    return jtu.tree_unflatten(treedef, keys)


def is_jitted(func: Callable):
    """Detect if a function is wrapped by jit or pmap."""
    return hasattr(func, "lower")


def has_nan(x: jax.Array) -> bool:
    """Check if the array has NaN values."""
    return jnp.isnan(x).any()


def tree_has_nan(tree: chex.ArrayTree) -> chex.ArrayTree:
    """Check if the pytree has NaN values."""
    return jtu.tree_map(has_nan, tree)


def invert_permutation(i: jax.Array) -> jax.Array:
    """Helper function that inverts a permutation array."""
    return jnp.empty_like(i).at[i].set(jnp.arange(i.size, dtype=i.dtype))


def _deepcopy(x):
    if isinstance(x, jax.Array):
        # we don't copy jax arrays, since they are immutable
        return x
    else:
        return copy.deepcopy(x)


def tree_deepcopy(tree: chex.ArrayTree) -> chex.ArrayTree:
    """Deep copy the pytree.

    Useful for mutable pytree structure like dict. The return also includes a deepcopy of these mutable structures.
    """
    return jtu.tree_map(_deepcopy, tree)


def right_shift_with_padding(
    x: chex.Array, shift: int, fill_value: None | chex.Scalar = None
):
    """Shift the array to the right with padding."""
    shifted_matrix = jnp.roll(x, shift=shift, axis=0)

    if fill_value is not None:
        padding = jnp.full_like(shifted_matrix[:shift], fill_value)
    else:
        padding = jnp.zeros_like(shifted_matrix[:shift])

    shifted_matrix = shifted_matrix.at[:shift].set(padding)

    return shifted_matrix


def sliding_window(arr, length, stride):
    """Slide a window over the fist axis of the array.

    Change shape from [T, ...] to [L, W, ...], where W = (T - L) // S + 1 is the number of windows.
    """
    starts = jnp.arange(0, arr.shape[0] - length + 1, stride)
    windows = jax.vmap(
        lambda start: jax.lax.dynamic_slice_in_dim(
            arr, start_index=start, slice_size=length, axis=0
        ),
    )(starts)

    return windows
