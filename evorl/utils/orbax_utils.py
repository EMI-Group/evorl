import jax.tree_util as jtu
import orbax.checkpoint as ocp
import chex


def save(path, state: chex.ArrayTree):
    ckpt = ocp.StandardCheckpointer()
    ckpt.save(path, args=ocp.args.StandardSave(state))


def load(path, state: chex.ArrayTree) -> chex.ArrayTree:
    """
        Args:
            path: checkpoint path
            state: the same structure as the saved state. Can be a dummy state 
                or its abstract_state by `jtu.tree_map(ocp.utils.to_shape_dtype_struct, state)`
    """
    ckpt = ocp.StandardCheckpointer()
    state = ckpt.restore(path, args=ocp.args.StandardRestore(state))
    return state
