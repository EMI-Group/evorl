import os
import logging
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import jax.tree_util as jtu
import chex
import orbax.checkpoint as ocp
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def to_shape_dtype_struct(value):
    """Create a restore template without expanding typed PRNG keys to uint32."""
    if isinstance(value, (jax.Array, jax.ShapeDtypeStruct)) and jax.dtypes.issubdtype(
        value.dtype, jax.dtypes.prng_key
    ):
        sharding = value.sharding
        if (
            isinstance(sharding, jax.sharding.NamedSharding)
            and len(sharding.spec) > value.ndim
            and all(axis is None for axis in sharding.spec[value.ndim :])
        ):
            sharding = jax.sharding.NamedSharding(
                sharding.mesh,
                jax.P(*sharding.spec[: value.ndim]),
                memory_kind=sharding.memory_kind,
            )
        return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding)
    return ocp.utils.to_shape_dtype_struct(value)


def save(path, state: chex.ArrayTree):
    """Save state to a file.

    Args:
        path: Checkpoint path.
        state: The state to be saved.
    """
    path = os.path.abspath(os.path.expanduser(path))

    state = filter_zero_size_arrays_on_save(state)

    with ocp.StandardCheckpointer() as ckpt:
        ckpt.save(path, state)


def load(path, state: chex.ArrayTree) -> chex.ArrayTree:
    """Load state from a file.

    Args:
        path: Checkpoint path
        state: The same structure as the saved state for restore. Can be a dummy state or its abstract_state by `jtu.tree_map(to_shape_dtype_struct, state)`

    Returns:
        The loaded state.
    """
    path = os.path.abspath(os.path.expanduser(path))
    abstract_state = jtu.tree_map(
        to_shape_dtype_struct, filter_zero_size_arrays_on_save(state)
    )

    with ocp.StandardCheckpointer() as ckpt:
        new_state = ckpt.restore(path, abstract_state)

    new_state = filter_zero_size_arrays_on_restore(state, new_state)

    return new_state


def filter_zero_size_arrays_on_save(
    tree: chex.ArrayTree,
) -> chex.ArrayTree:
    """Filter out zero-size arrays from the pytree."""

    def f(x):
        if isinstance(x, (jax.Array, np.ndarray, jax.ShapeDtypeStruct)) and x.size == 0:
            return None
        else:
            return x

    return jtu.tree_map(f, tree)


def filter_zero_size_arrays_on_restore(
    src_tree: chex.ArrayTree, dst_tree: chex.ArrayTree
) -> chex.ArrayTree:
    """Reconstruct zero-size leaves using the requested restore structure."""

    def f(src, dst):
        if isinstance(src, jax.ShapeDtypeStruct) and src.size == 0:
            return jnp.zeros(src.shape, dtype=src.dtype, device=src.sharding)
        if isinstance(src, (jax.Array, np.ndarray)) and src.size == 0:
            return src
        return dst

    return jtu.tree_map(f, src_tree, dst_tree)


class DummyCheckpointManager(ocp.AbstractCheckpointManager):
    """A dummy checkpoint manager that does nothing."""

    def __init__(self, directory="."):
        self._directory = Path(directory).expanduser().absolute()

    @property
    def directory(self):
        return self._directory

    def all_steps(self, read: bool = False) -> Sequence[int]:
        return []

    def latest_step(self) -> int | None:
        return None

    def best_step(self) -> int | None:
        return None

    def reload(self):
        pass

    def reached_preemption(self, step: int) -> bool:
        return False

    def should_save(self, step: int) -> bool:
        return False

    def delete(self, step: int):
        pass

    def item_metadata(self, step: int):
        return None

    def metadata(
        self, step: int | None = None
    ) -> ocp.metadata.RootMetadata | ocp.metadata.StepMetadata:
        if step is None:
            return ocp.metadata.RootMetadata()
        return ocp.metadata.StepMetadata()

    def metrics(self, step: int) -> Any | None:
        return None

    def wait_until_finished(self):
        pass

    def check_for_errors(self):
        pass

    def save(self, step, items=None, **kwargs):
        return False

    def restore(self, step, items=None, **kwargs):
        raise NotImplementedError("Checkpointing is disabled")

    def close(self):
        pass


class CheckpointManager(ocp.CheckpointManager):
    def save(self, step, items, **kwargs):
        args = ocp.args.StandardSave(filter_zero_size_arrays_on_save(items))
        return super().save(step, args=args, **kwargs)

    def restore(self, step, items, **kwargs):
        new_items = super().restore(
            step,
            args=ocp.args.StandardRestore(filter_zero_size_arrays_on_save(items)),
            **kwargs,
        )

        new_items = filter_zero_size_arrays_on_restore(items, new_items)
        return new_items


def setup_checkpoint_manager(config: DictConfig) -> ocp.AbstractCheckpointManager:
    """Setup checkpoint manager."""
    if config.checkpoint.enable:
        # Orbax only supports absolute checkpoint paths.
        output_dir = os.path.abspath(os.path.expanduser(config.output_dir))
        ckpt_path = os.path.join(output_dir, "checkpoints")
        ckpt_options = ocp.CheckpointManagerOptions(
            save_interval_steps=config.checkpoint.save_interval_steps,
            max_to_keep=config.checkpoint.max_to_keep,
        )
        logger.info(f"set checkpoint path: {ckpt_path}")
        checkpoint_manager = CheckpointManager(
            ckpt_path,
            options=ckpt_options,
            metadata=OmegaConf.to_container(
                config, resolve=True
            ),  # rescaled real config
        )
    else:
        checkpoint_manager = DummyCheckpointManager()

    return checkpoint_manager
