import chex
import jax
import jax.numpy as jnp

from evorl.types import PyTreeData


class ExponentialScheduleSpec(PyTreeData):
    init: float
    final: float
    decay: float


def weight_sum(x: jax.Array, w: jax.Array) -> jax.Array:
    """
    x: (n, ...)
    w: (n,)
    """
    chex.assert_equal_shape_prefix((x, w), 1)
    assert w.ndim == 1

    w = w.reshape(w.shape + (1,) * (x.ndim - 1))
    return jnp.sum(x * w, axis=0)
