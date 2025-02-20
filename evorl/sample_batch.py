import copy
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
    """Batched transitions w/ additional prefix axis as batch_axis.

    Could also be used as a trajectory.
    """

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

    def tree_replace(
        self, params: dict[str, jax.typing.ArrayLike | None]
    ) -> "PyTreeData":
        """Creates a new object with parameters set.

        Args:
            params: a dictionary of key value pairs to replace

        Returns:
            data clas with new values
        """
        new = self
        for k, v in params.items():
            new = _tree_replace(new, k.split("."), v)
        return new

    @property
    def T(self):  # pylint:disable=invalid-name
        return jtu.tree_map(lambda x: x.T, self)


def _tree_replace(
    base: PyTreeData,
    attr: Sequence[str],
    val: jax.typing.ArrayLike | None,
) -> PyTreeData:
    if not attr:
        return base

    # special case for List attribute
    if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
        lst = copy.deepcopy(getattr(base, attr[0]))

        for i, g in enumerate(lst):
            if not hasattr(g, attr[1]):
                continue
            v = val if not hasattr(val, "__iter__") else val[i]
            lst[i] = _tree_replace(g, attr[1:], v)

        return base.replace(**{attr[0]: lst})

    if len(attr) == 1:
        return base.replace(**{attr[0]: val})

    return base.replace(
        **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
    )


class Episode(PyTreeData):
    """The container for an episode trajectory."""

    trajectory: SampleBatch

    @property
    def valid_mask(self) -> chex.Array:
        return 1 - right_shift_with_padding(self.trajectory.dones, 1)
