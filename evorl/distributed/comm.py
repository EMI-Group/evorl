import jax
import jax.numpy as jnp
from jax._src.distributed import global_state
import chex
from typing import Optional, Sequence


def pmean(x, axis_name: Optional[str] = None):
    if axis_name is None:
        return x
    else:
        return jax.lax.pmean(x, axis_name)


def psum(x, axis_name: Optional[str] = None):
    if axis_name is None:
        return x
    else:
        return jax.lax.psum(x, axis_name)


def pmin(x, axis_name: Optional[str] = None):
    if axis_name is None:
        return x
    else:
        return jax.lax.pmin(x, axis_name)


def pmax(x, axis_name: Optional[str] = None):
    if axis_name is None:
        return x
    else:
        return jax.lax.pmax(x, axis_name)


def unpmap(x, axis_name: Optional[str] = None):
    """
        Only work for pmap(in_axes=0, out_axes=0)
        Return the first device's elements
    """
    if axis_name is None:
        return x
    else:
        return x[0]


def tree_pmean(tree: chex.ArrayTree, axis_name: Optional[str] = None):
    return jax.tree_map(lambda x: pmean(x, axis_name), tree)


def tree_unpmap(tree: chex.ArrayTree, axis_name: Optional[str] = None):
    return jax.tree_map(lambda x: unpmap(x, axis_name), tree)


def split_key_to_devices(key: chex.PRNGKey, devices: Sequence[jax.Device]):
    return jax.device_put_sharded(
        tuple(jax.random.split(key, len(devices))),
        devices
    )


def is_dist_initialized():
    # Note: global_state is a JAX internal API
    return global_state.coordinator_address is not None


def get_process_id():
    """
        Return the node id in multi-node distributed env.
    """
    if is_dist_initialized():
        return global_state.process_id
    else:
        return 0


def get_global_ranks():
    """
        Return the global rank for each device.
        Note: the return rank is already sharded.
    """

    num_local_devices = jax.local_device_count()

    process_id = get_process_id()
    ranks = process_id * num_local_devices + jnp.arange(
        num_local_devices, dtype=jnp.int32
    )
    ranks = jax.device_put_sharded(tuple(ranks), jax.local_devices())

    return ranks
