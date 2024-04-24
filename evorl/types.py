import copy
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from flax import struct
import chex

from typing import (
    Any, Mapping, Union, Dict, Optional, Sequence,
    Protocol
)
from typing_extensions import (
    dataclass_transform,  # pytype: disable=not-supported-yet
)
import dataclasses

from jax._src.util import safe_zip

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



class ObsPreprocessorFn(Protocol):
    def __call__(self, obs: chex.Array, *args: Any, **kwds: Any) -> chex.Array:
        return obs


@struct.dataclass
class PyTreeData:
    def __add__(self, o: Any) -> Any:
        return tree_map(lambda x, y: x + y, self, o)

    def __sub__(self, o: Any) -> Any:
        return tree_map(lambda x, y: x - y, self, o)

    def __mul__(self, o: Any) -> Any:
        return tree_map(lambda x: x * o, self)

    def __neg__(self) -> Any:
        return tree_map(lambda x: -x, self)

    def __truediv__(self, o: Any) -> Any:
        return tree_map(lambda x: x / o, self)

    def reshape(self, shape: Sequence[int]) -> Any:
        return tree_map(lambda x: x.reshape(shape), self)

    def select(self, o: Any, cond: jax.Array) -> Any:
        return tree_map(lambda x, y: (x.T * cond + y.T * (1 - cond)).T, self, o)

    def slice(self, beg: int, end: int) -> Any:
        return tree_map(lambda x: x[beg:end], self)

    def take(self, i, axis=0) -> Any:
        return tree_map(lambda x: jnp.take(x, i, axis=axis, mode='wrap'), self)

    def concatenate(self, *others: Any, axis: int = 0) -> Any:
        return tree_map(lambda *x: jnp.concatenate(x, axis=axis), self, *others)

    def index_set(
        self, idx: Union[jax.Array, Sequence[jax.Array]], o: Any
    ) -> Any:
        return tree_map(lambda x, y: x.at[idx].set(y), self, o)

    def index_sum(
        self, idx: Union[jax.Array, Sequence[jax.Array]], o: Any
    ) -> Any:
        return tree_map(lambda x, y: x.at[idx].add(y), self, o)

    # def vmap(self, in_axes=0, out_axes=0):
    #   """Returns an object that vmaps each follow-on instance method call."""

    #   # TODO: i think this is kinda handy, but maybe too clever?

    #   outer_self = self

    #   class VmapField:
    #     """Returns instance method calls as vmapped."""

    #     def __init__(self, in_axes, out_axes):
    #       self.in_axes = [in_axes]
    #       self.out_axes = [out_axes]

    #     def vmap(self, in_axes=0, out_axes=0):
    #       self.in_axes.append(in_axes)
    #       self.out_axes.append(out_axes)
    #       return self

    #     def __getattr__(self, attr):
    #       fun = getattr(outer_self.__class__, attr)
    #       # load the stack from the bottom up
    #       vmap_order = reversed(list(zip(self.in_axes, self.out_axes)))
    #       for in_axes, out_axes in vmap_order:
    #         fun = vmap(fun, in_axes=in_axes, out_axes=out_axes)
    #       fun = functools.partial(fun, outer_self)
    #       return fun

    #   return VmapField(in_axes, out_axes)

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
        return tree_map(lambda x: x.T, self)


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


class PyTreeDict(dict):
    """
        An easydict with pytree support
        Adapted from src: https://github.com/makinacorpus/easydict
    """

    def __init__(self, *args, **kwargs):
        d = dict(*args, **kwargs)

        for k, v in d.items():
            setattr(self, k, v)

        # # Class attributes
        # for k in self.__class__.__dict__.keys():
        #     if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #         setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(PyTreeDict, self).__setattr__(name, value)
        super(PyTreeDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(PyTreeDict, self).pop(k, d)
    
    def copy(self):
        d = super().copy() # dict
        return self.__class__(d)
    
    def replace(self, **d):
        clone = self.copy()
        clone.update(**d)
        return clone



jax.tree_util.register_pytree_node(
    PyTreeDict,
    lambda d: (tuple(d.values()), tuple(d.keys())),
    lambda keys, values: PyTreeDict(dict(safe_zip(keys, values)))
)

State = PyTreeDict

class EnvLike(Protocol):
    def reset(self, *args, **kwargs) -> Any:
        """Resets the environment to an initial state."""
        pass

    def step(self, *args, **kwargs) -> Any:
        """Run one timestep of the environment's dynamics."""
        pass


def pytree_field(*, lazy_init=False, pytree_node=True, **kwargs):
    """
        lazy_init: When set to True, the field will not be initialized in __init__,
            and we can use set_frozen_attr to set the value after __init__
    """
    if lazy_init:
        kwargs.update(dict(init=False, repr=False))
    return dataclasses.field(metadata={'pytree_node': pytree_node, 'lazy_init': lazy_init}, **kwargs)


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
                if field.metadata.get('lazy_init', False):
                    object.__setattr__(self, name, value)
                    return
                else:
                    raise dataclasses.FrozenInstanceError(
                        f"cannot assign to non-lazy_init field {name}")

        raise ValueError(
            f"field {name} not found in {self.__class__.__name__}")
