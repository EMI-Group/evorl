import chex
import jax


def get_tree_shape_prefix(tree: chex.ArrayTree, n_axes: int = 1) -> chex.Shape:
    """Get the shape of the leading axes (up to n_axes) of a pytree. This assumes all
    leaves have a common leading axes size (e.g. a common batch size)."""
    flat_tree, tree_def = jax.tree_util.tree_flatten(tree)
    leaf = flat_tree[0]
    leading_axis_shape = leaf.shape[0:n_axes]
    chex.assert_tree_shape_prefix(tree, leading_axis_shape)
    return leading_axis_shape


def get_buffer_size(buffer_state: chex.ArrayTree) -> int:
    """Utility to compute the total number of timesteps currently in the buffer state.

    Args:
        buffer_state (BufferStateTypes): the buffer state to compute the total timesteps for.

    Returns:
        int: the total number of timesteps in the buffer state.
    """
    # Ensure the buffer state is a valid buffer state.
    assert hasattr(buffer_state, "experience")
    assert hasattr(buffer_state, "current_index")
    assert hasattr(buffer_state, "is_full")

    b_size, t_size_max = get_tree_shape_prefix(buffer_state.experience, 2)
    t_size = jax.lax.cond(
        buffer_state.is_full,
        lambda: t_size_max,
        lambda: buffer_state.current_index,
    )
    timestep_count: int = b_size * t_size
    return timestep_count
