import jax
import jax.numpy as jnp
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


def split_key_to_devices(key: chex.PRNGKey, devices: Sequence[jax.Device]):
    num_devices = len(devices)
    return jax.device_put_sharded(
        jnp.split(jax.random.split(key, num_devices), num_devices),
        devices
    )
