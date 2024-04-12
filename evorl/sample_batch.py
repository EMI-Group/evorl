import jax.numpy as jnp
from jax.tree_util import tree_leaves
import chex
from flax import struct
from .types import (
    Reward, RewardDict, ExtraInfo, PyTreeData
)
from typing import (
    Any, Union, Optional
)

@struct.dataclass
class SampleBatch(PyTreeData):
    """
      Batched transitions w/ additional first axis as batch_axis.
      Could also be used as a trajectory.
    """
    # TODO: skip None in tree_map (should be work in native jax)
    obs: Optional[chex.ArrayTree] = None
    actions: Optional[chex.ArrayTree] = None
    rewards: Optional[Union[Reward, RewardDict]] = None
    next_obs: Optional[chex.Array] = None
    dones: Optional[chex.Array] = None
    extras: Optional[ExtraInfo] = None

    def __len__(self):
        return tree_leaves(self.obs)[0].shape[0]


def right_shift(arr: chex.Array, shift: int, pad_val=None) -> chex.Array:
    padding_shape = (shift, *arr.shape[1:])
    if pad_val is None:
        padding = jnp.zeros(padding_shape, dtype=arr.dtype)
    else:
        padding = jnp.full(padding_shape, pad_val, dtype=arr.dtype)
    return jnp.concatenate([padding, arr[:-shift]], axis=0)


@struct.dataclass
class Episode:
    trajectory: SampleBatch
    last_obs: chex.ArrayTree

    @property
    def valid_mask(self) -> chex.Array:
        return 1-right_shift(self.trajectory.dones, 1)