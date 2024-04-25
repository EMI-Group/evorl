from typing import Any, Mapping, Sequence, Optional
import jax.tree_util as jtu
import orbax.checkpoint as ocp
import chex
from orbax.checkpoint.composite_checkpoint_handler import CompositeArgs as Composite


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

class DummyCheckpointManager(ocp.AbstractCheckpointManager):
    def directory(self):
        return 'UwU'
    
    def all_steps(self, read: bool = False) -> Sequence[int]:
        return []
    
    def latest_step(self) -> Optional[int]:
        return None
    
    def best_step(self) -> int | None:
        return None
    
    def reload(self):
        pass
    
    def reached_preemption(self, step: int) -> bool:
        return True
    
    def should_save(self, step: int) -> bool:
        return False
    
    def delete(self, step: int):
        pass
    
    def item_metadata(self, step: int):
        return None
    
    def metadata(self) -> Mapping[str, Any]:
        return {}
    
    def metrics(self, step: int) -> Any | None:
        return None
    
    def wait_until_finished(self):
        pass
    
    def check_for_errors(self):
        pass

    def save(self,
             step: int,
             items=None,
             save_kwargs=None,
             metrics=None,
             force=False,
             args=None,
             ) -> bool:
        return True
    
    def restore(
        self,
        step,
        items = None,
        restore_kwargs= None,
        directory = None,
        args = None,
    ):
        raise NotImplementedError('UwU')
    
    def close(self):
        pass