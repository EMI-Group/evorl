from collections.abc import Sequence
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .types import ExtraInfo, PyTreeData, Reward, RewardDict
from .utils.jax_utils import right_shift_with_padding

__all__ = ["SampleBatch", "Episode"]


class SampleBatch(PyTreeData):
    """Data container for trajectory data."""

    obs: chex.ArrayTree | None = None
    actions: chex.ArrayTree | None = None
    rewards: Reward | RewardDict | None = None
    next_obs: chex.Array | None = None
    dones: chex.Array | None = None
    extras: ExtraInfo | None = None

    def __add__(self, o: Any) -> Any:
        return jtu.tree_map(lambda x, y: x + y, self, o)

    def __sub__(self, o: Any) -> Any:
        return jtu.tree_map(lambda x, y: x - y, self, o)

    def __mul__(self, o: Any) -> Any:
        return jtu.tree_map(lambda x: x * o, self)

    def __neg__(self) -> Any:
        return jtu.tree_map(lambda x: -x, self)

    def __truediv__(self, o: Any) -> Any:
        return jtu.tree_map(lambda x: x / o, self)

    def reshape(self, shape: Sequence[int]) -> Any:
        return jtu.tree_map(lambda x: x.reshape(shape), self)

    def select(self, o: Any, cond: jax.Array) -> Any:
        return jtu.tree_map(lambda x, y: (x.T * cond + y.T * (1 - cond)).T, self, o)

    def slice(self, beg: int, end: int) -> Any:
        return jtu.tree_map(lambda x: x[beg:end], self)

    def take(self, i, axis=0) -> Any:
        return jtu.tree_map(lambda x: jnp.take(x, i, axis=axis, mode="wrap"), self)

    def concatenate(self, *others: Any, axis: int = 0) -> Any:
        return jtu.tree_map(lambda *x: jnp.concatenate(x, axis=axis), self, *others)

    def index_set(self, idx: jax.Array | Sequence[jax.Array], o: Any) -> Any:
        return jtu.tree_map(lambda x, y: x.at[idx].set(y), self, o)

    def index_sum(self, idx: jax.Array | Sequence[jax.Array], o: Any) -> Any:
        return jtu.tree_map(lambda x, y: x.at[idx].add(y), self, o)

    @property
    def T(self):
        return jtu.tree_map(lambda x: x.T, self)


class Episode(PyTreeData):
    """The container for an episode trajectory."""

    trajectory: SampleBatch

    @property
    def valid_mask(self) -> chex.Array:
        return 1 - right_shift_with_padding(self.trajectory.dones, 1)
