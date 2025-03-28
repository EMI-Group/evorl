import os
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import jax.tree_util as jtu
import chex
import orbax.checkpoint as ocp
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

__all__ = [
    "save",
    "load",
    "setup_checkpoint_manager",
]


def save(path, state: chex.ArrayTree):
    """Save state to a file.

    Args:
        path: Checkpoint path.
        state: The state to be saved.
    """
    path = os.path.abspath(os.path.expanduser(path))

    with ocp.StandardCheckpointer() as ckpt:
        ckpt.save(path, state)


def load(path, state: chex.ArrayTree) -> chex.ArrayTree:
    """Load state from a file.

    Args:
        path: Checkpoint path
        state: The same structure as the saved state for restore. Can be a dummy state or its abstract_state by `jtu.tree_map(ocp.utils.to_shape_dtype_struct, state)`

    Returns:
        The loaded state.
    """
    path = os.path.abspath(os.path.expanduser(path))
    abstract_state = jtu.tree_map(ocp.utils.to_shape_dtype_struct, state)

    with ocp.StandardCheckpointer() as ckpt:
        new_state = ckpt.restore(path, abstract_state)
    return new_state


class DummyCheckpointManager(ocp.AbstractCheckpointManager):
    """A dummy checkpoint manager that does nothing."""

    def directory(self):
        return "UwU"

    def all_steps(self, read: bool = False) -> Sequence[int]:
        return []

    def latest_step(self) -> int | None:
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

    def save(
        self,
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
        items=None,
        restore_kwargs=None,
        directory=None,
        args=None,
    ):
        raise NotImplementedError("UwU")

    def close(self):
        pass


def setup_checkpoint_manager(config: DictConfig) -> ocp.CheckpointManager:
    """Setup checkpoint manager."""
    if config.checkpoint.enable:
        ckpt_options = ocp.CheckpointManagerOptions(
            save_interval_steps=config.checkpoint.save_interval_steps,
            max_to_keep=config.checkpoint.max_to_keep,
        )
        # Note: orbax only supports absolute path
        output_dir = os.path.abspath(os.path.expanduser(config.output_dir))
        ckpt_path = os.path.join(output_dir, "checkpoints")
        logger.info(f"set checkpoint path: {ckpt_path}")
        checkpoint_manager = ocp.CheckpointManager(
            ckpt_path,
            options=ckpt_options,
            metadata=OmegaConf.to_container(
                config, resolve=True
            ),  # rescaled real config
        )
    else:
        checkpoint_manager = DummyCheckpointManager()

    return checkpoint_manager
