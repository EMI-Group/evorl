"""Network parameter paths and NNX state operations."""

from collections.abc import Mapping

from flax import nnx
import jax
import jax.tree_util as jtu


def model_state_axes(
    model_state: nnx.GraphState,
    axes: Mapping[str, int | None] | nnx.StateAxes,
) -> nnx.GraphState:
    """Build a JAX vmap axis tree with the snapshot's static graph metadata.

    Args:
        model_state: Complete snapshot of a named `nnx.Dict` network group.
        axes: Axis for each network, or NNX filters specifying variable axes.
            For example, `{"actor": 0, "critic": None}` batches actors while
            sharing critics, and `nnx.StateAxes({nnx.Param: 0, ...: None})`
            batches weights while sharing non-parameter variables.

    Returns:
        An axis PyTree matching `model_state`, including its variable types
        and static graph metadata, suitable for ordinary `jax.vmap`.
    """

    def variable_axis(path, variable):
        names = tuple(getattr(key, "key", getattr(key, "name", None)) for key in path)
        axis = (
            axes.map_prefix(names, variable)
            if isinstance(axes, nnx.StateAxes)
            else axes[names[0]]
        )
        return jax.tree.map(lambda _: axis, variable)

    return jax.tree_util.tree_map_with_path(
        variable_axis,
        model_state,
        is_leaf=lambda value: isinstance(value, nnx.Variable),
    )


def is_layer_norm_layer(path: tuple[jtu.KeyEntry, ...]) -> bool:
    """Whether a parameter path belongs to a LayerNorm in the network layout.

    Recognize the NNX `layer_norms` collection and the legacy Linen `LayerNorm`
    module names used by parameter trees passed to evolutionary operators.
    """
    return any(
        isinstance(entry, jtu.DictKey)
        and isinstance(entry.key, str)
        and ("LayerNorm" in entry.key or entry.key == "layer_norms")
        for entry in path
    )
