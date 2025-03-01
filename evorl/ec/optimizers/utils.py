import chex
import jax
import jax.numpy as jnp
import optax

from evorl.types import PyTreeData


optimizer_map = dict(
    adam=optax.adam,
    sgd=optax.sgd,
    rmsprop=optax.rmsprop,
)


class ExponentialScheduleSpec(PyTreeData):
    """Specification for an exponential schedule for HyperParam."""

    init: float
    final: float
    decay: float


def weight_sum(x: jax.Array, w: jax.Array) -> jax.Array:
    """Weighted sum.

    Args:
        x: (n, ...)
        w: (n,)
    """
    chex.assert_equal_shape_prefix((x, w), 1)
    assert w.ndim == 1

    w = w.reshape(w.shape + (1,) * (x.ndim - 1))
    return jnp.sum(x * w, axis=0)
