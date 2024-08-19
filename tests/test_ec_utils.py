from evorl.utils.ec_utils import ParamVectorSpec
import flax.linen as nn
import jax
import jax.numpy as jnp
import chex


def test_ParamVectorSpec():
    model = nn.Dense(features=7)

    x = jnp.ones((2, 3))

    keys = jax.random.split(jax.random.PRNGKey(0), 5)

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

    batch_flat = param_spec.to_vector(batch_params)
    assert batch_flat.shape == (5, param_spec.vec_size)

    batch_recover = param_spec.to_tree(batch_flat)
    chex.assert_trees_all_close(batch_params, batch_recover)
