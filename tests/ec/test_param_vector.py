from evorl.utils.ec_utils import ParamVectorSpec
import flax.linen as nn
import jax
import jax.numpy as jnp
import chex
import pytest


def test_flax_parameters_roundtrip_through_vector_spec():
    model = nn.Dense(features=7)

    x = jnp.ones((2, 3))

    keys = jax.random.split(jax.random.key(0), 5)

    batch_params = [model.init(key, x) for key in keys]
    params = batch_params[0]
    batch_params = jax.tree_util.tree_map(
        lambda *x: jnp.stack(x, axis=0), *batch_params
    )

    param_spec = ParamVectorSpec(params)

    flat = param_spec.to_vector(params)
    assert flat.shape == (param_spec.vec_size,)
    recover = param_spec.to_tree(flat)
    chex.assert_trees_all_equal_shapes_and_dtypes(params, recover)
    chex.assert_trees_all_equal(params, recover)

    batch_flat = param_spec.to_vector(batch_params)
    assert batch_flat.shape == (5, param_spec.vec_size)

    batch_recover = param_spec.to_tree(batch_flat)
    chex.assert_trees_all_close(batch_params, batch_recover)


@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 4), (0,), (2, 0)])
def test_param_vector_spec_roundtrip_with_different_leaf_ranks(shape, batch_shape):
    params = {
        "first": jnp.zeros(shape),
        "second": (jnp.zeros(2, jnp.int32), None, jnp.zeros(2, jnp.float16)),
    }
    spec = ParamVectorSpec(params)

    def values(x):
        shape = batch_shape + x.shape
        size = 1
        for dim in shape:
            size *= dim
        return jnp.arange(size, dtype=jnp.float32).reshape(shape).astype(x.dtype)

    batched = jax.tree.map(values, params)
    flat = jax.jit(spec.to_vector)(batched)
    assert flat.shape == batch_shape + (spec.vec_size,)
    recovered = jax.jit(spec.to_tree)(flat)
    chex.assert_trees_all_equal(batched, recovered)
