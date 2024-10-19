import jax
import jax.tree_util as jtu

from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves

from evorl.sample_batch import SampleBatch


class ParamVectorSpec:
    def __init__(self, params):
        self._ndim = tree_leaves(params)[0].ndim
        flat, self.to_tree_fn = ravel_pytree(params)
        self.vec_size = flat.shape[0]
        self.to_vec_fn = lambda x: ravel_pytree(x)[0]

    def to_vector(self, x) -> jax.Array:
        """
        Return: (flat, to_tree_fn)
            see jax.flatten_util.ravel_pytree
        """
        leaves = tree_leaves(x)
        batch_ndim = leaves[0].ndim - self._ndim
        vmap_to_vector = self.to_vec_fn

        for _ in range(batch_ndim):
            vmap_to_vector = jax.vmap(vmap_to_vector)

        return vmap_to_vector(x)

    def to_tree(self, x) -> jax.Array:
        leaves = tree_leaves(x)
        batch_ndim = leaves[0].ndim - self._ndim
        vmap_to_tree = self.to_tree_fn

        for _ in range(batch_ndim):
            vmap_to_tree = jax.vmap(vmap_to_tree)

        return vmap_to_tree(x)


def flatten_pop_rollout_episode(trajectory: SampleBatch):
    """
    Flatten the trajectory from [#pop, T, B, ...] to [T, #pop*B, ...]
    """
    return jtu.tree_map(lambda x: jax.lax.collapse(x.swapaxes(0, 1), 1, 3), trajectory)
