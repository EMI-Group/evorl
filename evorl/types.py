import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Union

import chex
import jax.tree_util as jtu
from flax import struct
from jax.typing import ArrayLike, DTypeLike
from typing_extensions import dataclass_transform  # pytype: disable=not-supported-yet

Metrics = Mapping[str, chex.ArrayTree]
Observation = Union[chex.Array, Mapping[str, chex.Array]]
Action = Union[chex.Array, Mapping[str, chex.Array]]
Reward = Union[chex.Array, Mapping[str, chex.Array]]
Done = Union[chex.Array, Mapping[str, chex.Array]]
PolicyExtraInfo = Mapping[str, Any]
ExtraInfo = Mapping[str, Any]
RewardDict = Mapping[str, Reward]

LossDict = Mapping[str, chex.Array]

EnvInternalState = chex.ArrayTree

Params = chex.ArrayTree
ObsPreprocessorParams = Mapping[str, Any]
ActionPostprocessorParams = Mapping[str, Any]

AgentID = Any

ReplayBufferState = chex.ArrayTree


MISSING_REWARD = -1e10

Axis = Union[int, Sequence[int], None]


class ReductionFn(Protocol):
    def __call__(
        self,
        x: ArrayLike,
        axis: Axis = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: bool = False,
    ):
        pass


class ObsPreprocessorFn(Protocol):
    def __call__(self, obs: chex.Array, *args: Any, **kwds: Any) -> chex.Array:
        return obs


@jtu.register_pytree_node_class
class PyTreeDict(dict):
    """
    An easydict with pytree support
    """

    def __init__(self, *args, **kwargs):
        d = dict(*args, **kwargs)

        for k, v in d.items():
            setattr(self, k, v)

    @classmethod
    def _nested_convert(cls, obj):
        # currently only support dict, list, tuple (but not support their children class)
        if type(obj) is dict:
            return cls(obj)
        elif type(obj) is list:
            return list(cls._nested_convert(item) for item in obj)
        elif type(obj) is tuple:
            return tuple(cls._nested_convert(item) for item in obj)
        else:
            return obj

    def __setattr__(self, name, value):
        value = self._nested_convert(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super().pop(k, d)

    def copy(self):
        d = super().copy()  # dict
        return self.__class__(d)

    def replace(self, **d):
        clone = self.copy()
        clone.update(**d)
        return clone

    def tree_flatten(self):
        return tuple(self.values()), tuple(self.keys())

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(dict(zip(aux_data, children)))


@jtu.register_pytree_node_class
class State(PyTreeDict):
    pass


class EnvLike(Protocol):
    def reset(self, *args, **kwargs) -> Any:
        """Resets the environment to an initial state."""
        pass

    def step(self, *args, **kwargs) -> Any:
        """Run one timestep of the environment's dynamics."""
        pass


def pytree_field(*, lazy_init=False, pytree_node=True, **kwargs):
    """

    Args:
        lazy_init: When set to True, the field will not be initialized in `__init__()`, and we can use set_frozen_attr to set the value after `__init__`
        pytree_node: Setting to False will mark the field as static for pytree, that changing data in these fields will cause a re-jit of func.
    """
    if lazy_init:
        kwargs.update(dict(init=False, repr=False))

    metadata = {"pytree_node": pytree_node, "lazy_init": lazy_init}
    kwargs.setdefault("metadata", {}).update(metadata)

    return dataclasses.field(**kwargs)


@dataclass_transform(field_specifiers=(pytree_field,), kw_only_default=True)
class PyTreeNode:
    def __init_subclass__(cls, **kwargs):
        struct.dataclass(cls, **kwargs)

    def set_frozen_attr(self, name, value):
        """
        Force set attribute after __init__ of the dataclass
        """
        for field in dataclasses.fields(self):
            if field.name == name:
                if field.metadata.get("lazy_init", False):
                    object.__setattr__(self, name, value)
                    return
                else:
                    raise dataclasses.FrozenInstanceError(
                        f"cannot assign to non-lazy_init field {name}"
                    )

        raise ValueError(f"field {name} not found in {self.__class__.__name__}")


@dataclass_transform(field_specifiers=(pytree_field,), kw_only_default=True)
class PyTreeData:
    """
    Like PyTreeNode, but all fileds must be set at __init__, and not allow set_frozen_attr() method.
    Additionally, add some useful methods for PyTreeData.
    """

    def __init_subclass__(cls, **kwargs):
        struct.dataclass(cls, **kwargs)
