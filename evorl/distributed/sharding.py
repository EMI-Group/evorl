import jax
import jax.numpy as jnp
import chex

from jax.sharding import Sharding
from typing import Optional, Sequence, Union


def tree_device_put(tree: chex.ArrayTree, device):
    return jax.tree_map(lambda x: jax.device_put(x, device), tree)

def tree_device_get(tree: chex.ArrayTree, device=None):
    if device is None:
        device = jax.devices()[0]
    return tree_device_put(tree, device)