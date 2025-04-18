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
    """Define a metric field in `MetricBase`.

    Args:
        reduce_fn: A function to reduce the metric value across different devices. For example, `jax.mean`
        static: Whether the field is static related to pytree.

    Returns:
        A dataclass field.
    """
    metadata = {"static": static, "reduce_fn": reduce_fn}
    kwargs.setdefault("metadata", {}).update(metadata)

    return dataclasses.field(**kwargs)


class MetricBase(PyTreeData, kw_only=True):
    """Base class for all metrics."""

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
        """Convert the dataclass to native python structures recursively.

        The data in the metric object will be converted to local data types: list, tuple, dict, NamedTuple, etc. Jax array will be convert to numpy array,

        Returns:
            A converted dict.
        """
        return to_local_dict(self)


class WorkflowMetric(MetricBase):
    """Workflow metrics for RLWorkflow.

    Attributes:
        sampled_timesteps: The total number of sampled timesteps from environments.
        iterations: The total number of workflow iterations.
    """

    sampled_timesteps: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class TrainMetric(MetricBase):
    """Training metrics for RLWorkflow.

    Attributes:
        train_episode_return: The return of the training episode.
        loss: The loss value of the training step.
        raw_loss_dict: The raw loss dict of the training step.
    """

    # manually reduce in the step()
    train_episode_return: chex.Array | None = None

    # no need reduce_fn since it's already reduced in the step()
    loss: chex.Array = jnp.zeros(())
    raw_loss_dict: LossDict = metric_field(default_factory=PyTreeDict, reduce_fn=pmean)


class EvaluateMetric(MetricBase):
    """Evaluation metrics for RLWorkflow.

    Attributes:
        episode_returns: The return array of evaluation episodes.
        episode_lengths: The length array of evaluation episodes.
    """

    episode_returns: chex.Array = metric_field(reduce_fn=pmean)
    episode_lengths: chex.Array = metric_field(reduce_fn=pmean)


class ECWorkflowMetric(MetricBase):
    """Workflow metrics for ECWorkflow.

    Attributes:
        best_objective: The best objective value found so far.
        sampled_episodes: The total number of sampled episodes from environments..
        sampled_timesteps_m: The total number of sampled timesteps from environments, measured in millions.
        iterations: The total number of workflow iterations.
    """

    best_objective: chex.Array
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_timesteps_m: chex.Array = jnp.zeros((), dtype=jnp.float32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class MultiObjectiveECWorkflowMetric(MetricBase):
    """Workflow metrics for MultiObjectiveECWorkflow.

    Attributes:
        sampled_episodes: The total number of sampled episodes from environments..
        sampled_timesteps_m: The total number of sampled timesteps from environments, measured in millions.
        iterations: The total number of workflow iterations.
    """

    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_timesteps_m: chex.Array = jnp.zeros((), dtype=jnp.float32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class ECTrainMetric(MetricBase):
    """Training metrics for ECWorkflow.

    Attributes:
        objectives: The objective values for current step.
        ec_metrics: The extra metrics of the training step.
    """

    objectives: chex.Array
    ec_metrics: chex.ArrayTree


def to_local_dict(obj, *, dict_factory=dict):
    if not dataclasses.is_dataclass(obj):
        raise TypeError("to_local_dict() should be called on dataclass instances")
    return _to_local_dict_inner(obj, dict_factory)


def _to_local_dict_inner(obj, dict_factory):
    if dataclasses.is_dataclass(obj):
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
