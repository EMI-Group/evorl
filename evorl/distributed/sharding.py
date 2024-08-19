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
    sequential on the same gpu, parrallel on different gpu.
    """

    def shmap_f(state):
        # state: sharded state on single device
        # jax.debug.print("{}", state.env_state.obs.shape)
        return jax.lax.map(fn, state)

    return shard_map(
        shmap_f,
        mesh=sharding.mesh,
        in_specs=sharding.spec,
        out_specs=sharding.spec,
        # check_rep=False,
    )
