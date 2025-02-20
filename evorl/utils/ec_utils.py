import jax

from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves

__all__ = ["ParamVectorSpec"]


class ParamVectorSpec:
    """Save the structure of the parameters.

    Provide methods to convert between the original tree-like parameter and the flatten parameter vector.
    """

    def __init__(self, params):
        """Initialize the ParamVectorSpec.

        Args:
            params: Provide the structure of the parameters. It should be a single instance of model parameters instead of a batch of parameters.
        """
        self._ndim = tree_leaves(params)[0].ndim
        flat, self.to_tree_fn = ravel_pytree(params)
        self.vec_size = flat.shape[0]
        self.to_vec_fn = lambda x: ravel_pytree(x)[0]

    def to_vector(self, x) -> jax.Array:
        """Convert the original params to flatten params.

        see `jax.flatten_util.ravel_pytree`

        Args:
            x: The original params.

        Returns:
            Flatten params.
        """
        leaves = tree_leaves(x)
        batch_ndim = leaves[0].ndim - self._ndim
        vmap_to_vector = self.to_vec_fn

        for _ in range(batch_ndim):
            vmap_to_vector = jax.vmap(vmap_to_vector)

        return vmap_to_vector(x)

    def to_tree(self, x) -> jax.Array:
        """Convert the flatten params to the original params.

        Args:
            x: The flatten params.

        Returns:
            The original params.
        """
        leaves = tree_leaves(x)
        batch_ndim = leaves[0].ndim - self._ndim
        vmap_to_tree = self.to_tree_fn

        for _ in range(batch_ndim):
            vmap_to_tree = jax.vmap(vmap_to_tree)

        return vmap_to_tree(x)
