import jax
import jax.numpy as jnp


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


def all_gather(x, axis_name: str | None = None, **kwargs):
    """All-gather the data across all devices."""
    if axis_name is None:
        return x
    else:
        return jax.lax.all_gather(x, axis_name, **kwargs)


def is_dist_initialized():
    """Whether the JAX's distributed setting is initialized."""
    return jax.distributed.is_initialized()


def get_process_id():
    """Return the node id in multi-node distributed env."""
    if is_dist_initialized():
        return jax.process_index()
    else:
        return 0


def get_global_ranks(axis_name: str) -> jax.Array:
    """Return the global rank for each device.

    Returns:
        The sharded ranks across devices. Each device has a unique rank.
    """
    num_local_devices = jax.sharding.get_abstract_mesh().shape[axis_name]

    process_id = get_process_id()
    ranks = process_id * num_local_devices + jnp.arange(
        num_local_devices, dtype=jnp.int32
    )
    ranks = jax.device_put(ranks, jax.P(axis_name))

    return ranks
