import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_leaves, tree_unflatten
from jax.flatten_util import ravel_pytree
from collections import namedtuple
from typing import Callable, Tuple
from flax import struct
    
class ParamVectorSpec:
    def __init__(self, params):
        self._ndim = tree_leaves(params)[0].ndim
        flat, self.to_tree_fn = ravel_pytree(params)
        self.vec_size = flat.shape[0]
        self.to_vec_fn = lambda x: ravel_pytree(x)[0]

    def to_vector(self, x)-> Tuple[jax.Array, Callable]:
        """
            Return: (flat, to_tree_fn)
                see jax.flatten_util.ravel_pytree
        """
        leaves = tree_leaves(x)
        batch_ndim = leaves[0].ndim - self._ndim
        vmap_to_tree = self.to_vec_fn

        for _ in range(batch_ndim):
            vmap_to_tree = jax.vmap(vmap_to_tree)
        
        flat = vmap_to_tree(x)

        vmap_to_tree = self.to_tree_fn
        for _ in range(batch_ndim):
            vmap_to_tree = jax.vmap(vmap_to_tree)

        return flat, vmap_to_tree