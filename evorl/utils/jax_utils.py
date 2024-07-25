import os
import re
from functools import partial
from collections.abc import Callable, Iterable, Sequence

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


def tree_concat(nest1, nest2, axis=0):
    return jtu.tree_map(lambda x, y: jnp.concatenate([x, y], axis=axis), nest1, nest2)


def tree_stop_gradient(nest: chex.ArrayTree) -> chex.ArrayTree:
    return jtu.tree_map(jax.lax.stop_gradient, nest)


def tree_astype(tree, dtype):
    return jtu.tree_map(lambda x: x.astype(dtype), tree)


def tree_last(tree):
    return jtu.tree_map(lambda x: x[-1], tree)


def scan_and_mean(*args, **kwargs):
    """
    usage: same like `jax.lax.scan`, bug the scan results will be averaged.
    """
    last_carry, ys = jax.lax.scan(*args, **kwargs)
    return last_carry, jtu.tree_map(lambda x: x.mean(axis=0), ys)


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


_vmap_rng_split_fn = jax.vmap(jax.random.split, in_axes=(0, None), out_axes=1)


def vmap_rng_split(key: jax.Array, num: int = 2) -> jax.Array:
    # batched_key [B, 2] -> batched_keys [num, B, 2]
    chex.assert_shape(key, (None, 2))
    return _vmap_rng_split_fn(key, num)


def rng_split(key: jax.Array, num: int = 2) -> jax.Array:
    """
    Unified Version of `jax.random.split` for both single key and batched keys.
    """
    if key.ndim == 1:
        chex.assert_shape(key, (2,))
        return jax.random.split(key, num)
    else:
        return vmap_rng_split(key, num)


def is_jitted(func):
    """
    Detect if a function is wrapped by jit or pmap.
    """
    return hasattr(func, "lower")
