import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map
import chex
from flax import struct
from typing import (
    Any, Union, Tuple, Optional, Sequence, Callable
)

from .types import LossDict
from .distributed import pmean, psum
import dataclasses

def metricfield(*, reduce_fn: Callable[[chex.Array, Optional[str]], chex.Array] = None, pytree_node=True, **kwargs):
    return dataclasses.field(metadata={'pytree_node': pytree_node, 'reduce_fn': reduce_fn}, **kwargs)

# TODO: use kw_only=True when jax support it


class MetricBase(struct.PyTreeNode):
    def all_reduce(self, pmap_axis_name: Optional[str] = None):
        field_dict = {}
        for field in dataclasses.fields(self):
            reduce_fn = field.metadata.get('reduce_fn', None)
            value = getattr(self, field.name)
            if pmap_axis_name is not None and isinstance(reduce_fn, Callable):
                value = reduce_fn(value, pmap_axis_name)
                field_dict[field.name] = value

        if len(field_dict) == 0:
            return self

        return self.replace(**field_dict)


class WorkflowMetric(MetricBase):
    sampled_timesteps: chex.Array = metricfield(
        default=jnp.zeros((), dtype=jnp.int32), reduce_fn=psum)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.int32)


class TrainMetric(MetricBase):
    train_episode_return: chex.Array = metricfield(
        default=jnp.zeros(()), reduce_fn=pmean)
    # no need reduce_fn since it's already reduced in the step()
    loss: chex.Array = jnp.zeros((), dtype=jnp.int32)
    raw_loss_dict: LossDict = metricfield(default_factory=dict)


class EvaluateMetric(MetricBase):
    discount_returns: chex.Array = metricfield(reduce_fn=pmean)
    episode_lengths: chex.Array = metricfield(reduce_fn=pmean)
