from collections.abc import Callable

import chex
import jax
from jax.experimental.shard_map import shard_map


def tree_device_put(tree: chex.ArrayTree, device):
    return jax.tree_map(lambda x: jax.device_put(x, device), tree)


def tree_device_get(tree: chex.ArrayTree, device=None):
    if device is None:
        device = jax.devices()[0]
    return tree_device_put(tree, device)


def parallel_map(fn: Callable, sharding):
    """
    Sequential execution on different gpu.

    Args:
        fn: function to be executed, only positional arguments are supported
        sharding: JAX sharding object
    """

    def _f(carry):
        return fn(*carry)

    def shmap_f(*args):
        # state: sharded state on single device

        return jax.lax.map(_f, args)

    return shard_map(
        shmap_f,
        mesh=sharding.mesh,
        in_specs=sharding.spec,
        out_specs=sharding.spec,
        # check_rep=False,
    )
