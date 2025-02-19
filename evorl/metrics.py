import dataclasses
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np

from .distributed import pmean
from .types import LossDict, PyTreeData, PyTreeDict


def metric_field(
    *,
    reduce_fn: Callable[[chex.Array, str | None], chex.Array] = None,
    static=False,
    **kwargs,
):
    """"""
    metadata = {"static": static, "reduce_fn": reduce_fn}
    kwargs.setdefault("metadata", {}).update(metadata)

    return dataclasses.field(**kwargs)


class MetricBase(PyTreeData, kw_only=True):
    def all_reduce(self, pmap_axis_name: str | None = None):
        field_dict = {}
        for field in dataclasses.fields(self):
            reduce_fn = field.metadata.get("reduce_fn", None)
            value = getattr(self, field.name)
            if pmap_axis_name is not None and isinstance(reduce_fn, Callable):
                value = reduce_fn(value, pmap_axis_name)
                field_dict[field.name] = value

        if len(field_dict) == 0:
            return self

        return self.replace(**field_dict)

    def to_local_dict(self):
        """Convert all dataclass to dict and convert
        jax array and numpy array to python list
        """
        return to_local_dict(self)


class WorkflowMetric(MetricBase):
    sampled_timesteps: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class TrainMetric(MetricBase):
    # manually reduce in the step()
    train_episode_return: chex.Array | None = None

    # no need reduce_fn since it's already reduced in the step()
    loss: chex.Array = jnp.zeros(())
    raw_loss_dict: LossDict = metric_field(default_factory=PyTreeDict, reduce_fn=pmean)


class EvaluateMetric(MetricBase):
    episode_returns: chex.Array = metric_field(reduce_fn=pmean)
    episode_lengths: chex.Array = metric_field(reduce_fn=pmean)


def _is_dataclass_instance(obj):
    """Returns True if obj is an instance of a dataclass."""
    return hasattr(type(obj), "__dataclass_fields__")


def to_local_dict(obj, *, dict_factory=dict):
    if not _is_dataclass_instance(obj):
        raise TypeError("to_local_dict() should be called on dataclass instances")
    return _to_local_dict_inner(obj, dict_factory)


def _to_local_dict_inner(obj, dict_factory):
    if _is_dataclass_instance(obj):
        result = []
        for f in dataclasses.fields(obj):
            value = _to_local_dict_inner(getattr(obj, f.name), dict_factory)
            result.append((f.name, value))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict()
        # method, because:
        # - it does not recurse in to the namedtuple fields and
        #   convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The main
        #   use case here is json.dumps, and it handles converting
        #   namedtuples to lists.  Admittedly we're losing some
        #   information here when we produce a json list instead of a
        #   dict.  Note that if we returned dicts here instead of
        #   namedtuples, we could no longer call asdict() on a data
        #   structure where a namedtuple was used as a dict key.

        return type(obj)(*[_to_local_dict_inner(v, dict_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_to_local_dict_inner(v, dict_factory) for v in obj)
    elif isinstance(obj, PyTreeDict):
        return {
            _to_local_dict_inner(k, dict_factory): _to_local_dict_inner(v, dict_factory)
            for k, v in obj.items()
        }
    elif isinstance(obj, dict):
        return type(obj)(
            (
                _to_local_dict_inner(k, dict_factory),
                _to_local_dict_inner(v, dict_factory),
            )
            for k, v in obj.items()
        )
    else:
        if isinstance(obj, jax.Array):
            return np.array(obj)
        else:
            return obj
