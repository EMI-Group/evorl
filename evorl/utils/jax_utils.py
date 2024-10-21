import os
from collections.abc import Iterable, Sequence, Callable
from functools import partial
import math

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def disable_gpu_preallocation():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def optimize_gpu_utilization():
    xla_flags = os.getenv("XLA_FLAGS", "")
    # print(f"current XLA_FLAGS: {xla_flags}")
    if len(xla_flags) > 0:
        xla_flags = xla_flags + " "
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
    return jtu.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), nest)


def tree_ones_like(nest: chex.ArrayTree, dtype=None) -> chex.ArrayTree:
    return jtu.tree_map(lambda x: jnp.ones(x.shape, dtype or x.dtype), nest)


def tree_concat(nest1: chex.ArrayTree, nest2: chex.ArrayTree, axis: int = 0):
    return jtu.tree_map(lambda x, y: jnp.concatenate([x, y], axis=axis), nest1, nest2)


def tree_stop_gradient(nest: chex.ArrayTree) -> chex.ArrayTree:
    return jtu.tree_map(jax.lax.stop_gradient, nest)


def tree_astype(tree: chex.ArrayTree, dtype):
    return jtu.tree_map(lambda x: x.astype(dtype), tree)


def tree_last(tree: chex.ArrayTree):
    return jtu.tree_map(lambda x: x[-1], tree)


def tree_get(tree: chex.ArrayTree, idx_or_slice):
    return jtu.tree_map(lambda x: x[idx_or_slice], tree)


def tree_set(
    src: chex.ArrayTree,
    target: chex.ArrayTree,
    idx_or_slice,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | None = None,
):
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
    """
    usage: same like `jax.lax.scan`, but the scan results will be averaged.
    """
    last_carry, ys = jax.lax.scan(*args, **kwargs)
    return last_carry, jtu.tree_map(lambda x: x.mean(axis=0), ys)


def scan_and_last(*args, **kwargs):
    """
    usage: same like `jax.lax.scan`, but return the last scan iteration results.
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
    """
    A decorator for `jax.jit` with arguments.

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
    """
    A decorator for `jax.pmap` with arguments.
    """
    return partial(
        jax.pmap,
        axis_name,
        static_broadcasted_argnums=static_broadcasted_argnums,
        donate_argnums=donate_argnums,
        **kwargs,
    )


def vmap_rng_split(key: chex.PRNGKey, num: int = 2) -> chex.PRNGKey:
    # batched_key [B, 2] -> batched_keys [num, B, 2]
    chex.assert_shape(key, (..., 2))

    rng_split_fn = jax.random.split

    for _ in range(key.ndim - 1):
        rng_split_fn = jax.vmap(rng_split_fn, in_axes=(0, None), out_axes=1)

    return rng_split_fn(key, num)


def rng_split(key: chex.PRNGKey, num: int = 2) -> chex.PRNGKey:
    """
    Unified Version of `jax.random.split` for both single key and batched keys.
    """
    if key.ndim == 1:
        chex.assert_shape(key, (2,))
        return jax.random.split(key, num)
    else:
        return vmap_rng_split(key, num)


def rng_split_by_shape(key: chex.PRNGKey, shape: tuple[int]) -> chex.PRNGKey:
    chex.assert_shape(key, (2,))
    keys = jax.random.split(key, math.prod(shape))
    return jnp.reshape(keys, shape + (2,))


def rng_split_like_tree(key: chex.PRNGKey, target: chex.ArrayTree) -> chex.ArrayTree:
    """
    Returns:
        A tree that each like has a single key.
    """
    treedef = jax.tree_structure(target)
    keys = jax.random.split(key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)


def is_jitted(func: Callable):
    """
    Detect if a function is wrapped by jit or pmap.
    """
    return hasattr(func, "lower")


def has_nan(x: jax.Array) -> bool:
    return jnp.isnan(x).any()


def tree_has_nan(tree: chex.ArrayTree) -> chex.ArrayTree:
    return jtu.tree_map(has_nan, tree)


def invert_permutation(i: jax.Array) -> jax.Array:
    """Helper function that inverts a permutation array."""
    return jnp.empty_like(i).at[i].set(jnp.arange(i.size, dtype=i.dtype))


def right_shift_with_padding(
    x: chex.Array, shift: int, fill_value: None | chex.Scalar = None
):
    shifted_matrix = jnp.roll(x, shift=shift, axis=0)

    if fill_value is not None:
        padding = jnp.full_like(shifted_matrix[:shift], fill_value)
    else:
        padding = jnp.zeros_like(shifted_matrix[:shift])

    shifted_matrix = shifted_matrix.at[:shift].set(padding)

    return shifted_matrix
