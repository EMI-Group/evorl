import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import chex

import copy
from typing import Sequence, Dict
from .types import (
    Reward, RewardDict, ExtraInfo, PyTreeData
)
from typing import (
    Any, Union, Optional
)

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
        return jtu.tree_map(lambda x: jnp.take(x, i, axis=axis, mode='wrap'), self)

    def concatenate(self, *others: Any, axis: int = 0) -> Any:
        return jtu.tree_map(lambda *x: jnp.concatenate(x, axis=axis), self, *others)

    def index_set(
        self, idx: Union[jax.Array, Sequence[jax.Array]], o: Any
    ) -> Any:
        return jtu.tree_map(lambda x, y: x.at[idx].set(y), self, o)

    def index_sum(
        self, idx: Union[jax.Array, Sequence[jax.Array]], o: Any
    ) -> Any:
        return jtu.tree_map(lambda x, y: x.at[idx].add(y), self, o)

    def tree_replace(
        self, params: Dict[str, Optional[jax.typing.ArrayLike]]
    ) -> 'PyTreeData':
        """Creates a new object with parameters set.

        Args:
          params: a dictionary of key value pairs to replace

        Returns:
          data clas with new values

        Example:
          If a system has 3 links, the following code replaces the mass
          of each link in the System:
          >>> sys = sys.tree_replace(
          >>>     {'link.inertia.mass', jnp.array([1.0, 1.2, 1.3])})
        """
        new = self
        for k, v in params.items():
            new = _tree_replace(new, k.split('.'), v)
        return new

    @property
    def T(self):  # pylint:disable=invalid-name
        return jtu.tree_map(lambda x: x.T, self)


def _tree_replace(
    base: PyTreeData,
    attr: Sequence[str],
    val: Optional[jax.typing.ArrayLike],
) -> PyTreeData:
    """Sets attributes in a struct.dataclass with values."""
    if not attr:
        return base

    # special case for List attribute
    if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
        lst = copy.deepcopy(getattr(base, attr[0]))

        for i, g in enumerate(lst):
            if not hasattr(g, attr[1]):
                continue
            v = val if not hasattr(val, '__iter__') else val[i]
            lst[i] = _tree_replace(g, attr[1:], v)

        return base.replace(**{attr[0]: lst})

    if len(attr) == 1:
        return base.replace(**{attr[0]: val})

    return base.replace(
        **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
    )


def right_shift(arr: chex.Array, shift: int, pad_val=None) -> chex.Array:
    padding_shape = (shift, *arr.shape[1:])
    if pad_val is None:
        padding = jnp.zeros(padding_shape, dtype=arr.dtype)
    else:
        padding = jnp.full(padding_shape, pad_val, dtype=arr.dtype)
    return jnp.concatenate([padding, arr[:-shift]], axis=0)


class Episode(PyTreeData):
    trajectory: SampleBatch
    last_obs: chex.ArrayTree

    @property
    def valid_mask(self) -> chex.Array:
        return 1-right_shift(self.trajectory.dones, 1)
    
