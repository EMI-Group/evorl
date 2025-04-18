"""Common type annotations and data structures."""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Union, TypeVar
from typing_extensions import dataclass_transform  # pytype: disable=not-supported-yet

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

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

Axis = int | None | Sequence[Any]

MISSING_REWARD = -1e10


# __all__ = [
#     "pytree_field",
#     "dataclass",
#     "PyTreeDict",
#     "State",
#     "PyTreeNode",
#     "PyTreeData",
# ]


class PyTreeArrayMixin:
    """batch operate pytree with jax.Array.

    It assumes all arrays have the same head shape.
    """

    def __add__(self, o: chex.ArrayTree) -> chex.ArrayTree:
        return jtu.tree_map(lambda x, y: x + y, self, o)

    def __sub__(self, o: chex.ArrayTree) -> chex.ArrayTree:
        return jtu.tree_map(lambda x, y: x - y, self, o)

    def __mul__(self, o: chex.ArrayTree) -> chex.ArrayTree:
        return jtu.tree_map(lambda x: x * o, self)

    def __neg__(self) -> chex.ArrayTree:
        return jtu.tree_map(lambda x: -x, self)

    def __truediv__(self, o: chex.ArrayTree) -> chex.ArrayTree:
        return jtu.tree_map(lambda x: x / o, self)

    def reshape(self, shape: Sequence[int]) -> chex.ArrayTree:
        return jtu.tree_map(lambda x: x.reshape(shape), self)

    def slice(self, beg: int, end: int, strides=None) -> chex.ArrayTree:
        return jtu.tree_map(lambda x: x[beg:end:strides], self)

    def take(self, i, axis=0) -> chex.ArrayTree:
        return jtu.tree_map(lambda x: jnp.take(x, i, axis=axis, mode="wrap"), self)

    def concatenate(self, *others: chex.ArrayTree, axis: int = 0) -> chex.ArrayTree:
        return jtu.tree_map(lambda *x: jnp.concatenate(x, axis=axis), self, *others)

    def index_set(
        self, idx: jax.Array | Sequence[jax.Array], o: chex.ArrayTree
    ) -> chex.ArrayTree:
        return jtu.tree_map(lambda x, y: x.at[idx].set(y), self, o)

    def index_sum(
        self, idx: jax.Array | Sequence[jax.Array], o: chex.ArrayTree
    ) -> chex.ArrayTree:
        return jtu.tree_map(lambda x, y: x.at[idx].add(y), self, o)

    @property
    def T(self):
        return jtu.tree_map(lambda x: x.T, self)


@jtu.register_pytree_node_class
class PyTreeDict(dict):
    """An easydict with pytree support."""

    def __init__(self, *args, **kwargs):
        d = dict(*args, **kwargs)

        for k, v in d.items():
            setattr(self, k, v)

    @classmethod
    def _nested_convert(cls, obj):
        # Currently, only support dict, list, tuple (not include their children classes)
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
    """A general State class.

    An alias of PyTreeDict. This class is specfically used for `Workflow` state.
    """

    pass


def pytree_field(*, lazy_init=False, static=False, **kwargs):
    """Define a pytree field in our dataclass.

    Args:
        lazy_init: When set to True, the field will not be initialized in `__init__()`, and we can use set_frozen_attr to set the value after `__init__`
        static: Setting to False will mark the field as static for pytree, that changing data in these fields will cause a re-jit of func.

    Returns:
        A dataclass field.
    """
    if lazy_init:
        kwargs.update(init=False, repr=False)

    metadata = {"static": static, "lazy_init": lazy_init}
    kwargs.setdefault("metadata", {}).update(metadata)

    return dataclasses.field(**kwargs)


_T = TypeVar("T")


@dataclass_transform(field_specifiers=(pytree_field,))  # type: ignore[literal-required]
def dataclass(clz: _T, *, pure_data=False, **kwargs) -> _T:
    if "frozen" not in kwargs.keys():
        kwargs["frozen"] = True
    data_clz = dataclasses.dataclass(**kwargs)(clz)  # type: ignore
    meta_fields = []
    data_fields = []
    for field_info in dataclasses.fields(data_clz):
        is_static = field_info.metadata.get("static", False)
        if is_static:
            meta_fields.append(field_info.name)
        else:
            data_fields.append(field_info.name)

    def replace(self, **updates):
        """Returns a new object replacing the specified fields with new values."""
        return dataclasses.replace(self, **updates)

    data_clz.replace = replace

    if pure_data and hasattr(jax.tree_util, "register_dataclass"):
        # use the optimized C++ dataclass builtin (jax>=0.4.26)
        jax.tree_util.register_dataclass(data_clz, data_fields, meta_fields)
    else:

        def iterate_clz(x):
            meta = tuple(getattr(x, name) for name in meta_fields)
            data = tuple(getattr(x, name) for name in data_fields)
            return data, meta

        def iterate_clz_with_keys(x):
            meta = tuple(getattr(x, name) for name in meta_fields)
            data = tuple(
                (jax.tree_util.GetAttrKey(name), getattr(x, name))
                for name in data_fields
            )
            return data, meta

        def clz_from_iterable(meta, data):
            meta_args = tuple(zip(meta_fields, meta))
            data_args = tuple(zip(data_fields, data))
            kwargs = dict(meta_args + data_args)
            return data_clz(**kwargs)

        jax.tree_util.register_pytree_with_keys(
            data_clz,
            iterate_clz_with_keys,
            clz_from_iterable,
            iterate_clz,
        )

    return data_clz  # type: ignore


@dataclass_transform(field_specifiers=(pytree_field,), kw_only_default=True)
class PyTreeNode:
    """A pytree dataclass for Node."""

    def __init_subclass__(cls, kw_only=True, **kwargs):
        dataclass(cls, pure_data=False, kw_only=kw_only, **kwargs)

    def set_frozen_attr(self, name, value):
        """Force set attribute after __init__ of the dataclass."""
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
    """A pytree dataclass for Data.

    Like `PyTreeNode`, but all fileds must be set at __init__, and not allow set_frozen_attr() method.
    """

    def __init_subclass__(cls, kw_only=True, **kwargs):
        dataclass(cls, pure_data=True, kw_only=kw_only, **kwargs)
