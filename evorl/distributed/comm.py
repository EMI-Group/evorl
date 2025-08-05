from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax._src.distributed import global_state


def pmean(x, axis_name: str | None = None):
    if axis_name is None:
        return x
    else:
        return jax.lax.pmean(x, axis_name)


def psum(x, axis_name: str | None = None):
    if axis_name is None:
        return x
    else:
        return jax.lax.psum(x, axis_name)


def pmin(x, axis_name: str | None = None):
    if axis_name is None:
        return x
    else:
        return jax.lax.pmin(x, axis_name)


def pmax(x, axis_name: str | None = None):
    if axis_name is None:
        return x
    else:
        return jax.lax.pmax(x, axis_name)


def _unpmap(x, axis_name: str | None = None):
    # Only work for pmap(in_axes=0, out_axes=0)
    # Return the first device's elements
    if axis_name is None:
        return x
    else:
        return x[0]


def unpmap(tree: chex.ArrayTree, axis_name: str | None = None):
    return jtu.tree_map(lambda x: _unpmap(x, axis_name), tree)


def all_gather(x, axis_name: str | None = None, **kwargs):
    """All-gather the data across all devices."""
    if axis_name is None:
        return x
    else:
        return jax.lax.all_gather(x, axis_name, **kwargs)


def split_key_to_devices(key: chex.PRNGKey, devices: Sequence[jax.Device]):
    """Split the key to each device."""
    return jax.device_put_sharded(tuple(jax.random.split(key, len(devices))), devices)


def is_dist_initialized():
    """Whether the JAX's distributed setting is initialized."""
    # Note: global_state is a JAX internal API, which is not stable.
    return global_state.coordinator_address is not None


def get_process_id():
    """Return the node id in multi-node distributed env."""
    if is_dist_initialized():
        return global_state.process_id
    else:
        return 0


def get_global_ranks():
    """Return the global rank for each device.

    Returns:
        The sharded ranks across devices. Each device has a unique rank.
    """
    num_local_devices = jax.local_device_count()

    process_id = get_process_id()
    ranks = process_id * num_local_devices + jnp.arange(
        num_local_devices, dtype=jnp.int32
    )
    ranks = jax.device_put_sharded(tuple(ranks), jax.local_devices())

    return ranks
