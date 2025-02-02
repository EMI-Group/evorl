from collections.abc import Callable

import chex
import jax
from jax.experimental.shard_map import shard_map


def tree_device_put(tree: chex.ArrayTree, device_or_sharding):
    return jax.tree_map(lambda x: jax.device_put(x, device_or_sharding), tree)


def tree_device_get(tree: chex.ArrayTree, device=None):
    if device is None:
        device = jax.devices()[0]
    return tree_device_put(tree, device)


def shmap_vmap(fn: Callable, mesh, in_specs, out_specs, **kwargs):
    def shmap_f(*args):
        return jax.vmap(fn)(*args)

    return shard_map(
        shmap_f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kwargs
    )


def shmap_map(fn: Callable, mesh, in_specs, out_specs, **kwargs):
    """
    Sequential execution on different gpu.

    Args:
        fn: function to be executed, only positional arguments are supported
        sharding: JAX sharding object
    """

    def g(carry):
        return fn(*carry)

    def shmap_f(*args):
        return jax.lax.map(g, args)

    return shard_map(
        shmap_f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kwargs
    )
