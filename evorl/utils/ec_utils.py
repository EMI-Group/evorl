import jax

from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves


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


def convert_so_train_metrics(train_metrics):
    train_metrics_dict = train_metrics.to_local_dict()

    train_metrics_dict["objectives"] = train_metrics.objectives.tolist()
