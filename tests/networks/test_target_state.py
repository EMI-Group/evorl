"""Target updates distinguish weights from model bookkeeping."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from evorl.utils.rl_toolkits import soft_target_update


class StatefulModel(nnx.Module):
    def __init__(self, weight, count):
        self.weight = nnx.Param(jnp.float32(weight))
        self.count = nnx.Variable(jnp.int32(count))


def test_soft_update_interpolates_parameters_and_preserves_target_state(jit):
    target = nnx.state(StatefulModel(2, 3))
    source = nnx.state(StatefulModel(6, 9))
    update = jax.jit(soft_target_update) if jit else soft_target_update
    result = update(target, source, 0.25)
    np.testing.assert_array_equal(result["weight"][...], 3)
    np.testing.assert_array_equal(result["count"][...], 3)
    np.testing.assert_array_equal(target["weight"][...], 2)
    np.testing.assert_array_equal(source["weight"][...], 6)
