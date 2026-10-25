import json

from flax import nnx
import jax
import jax.numpy as jnp
import chex
import pytest

from evorl.algorithms.ec.so.cmaes import _get_std_statistics
from evorl.algorithms.erl.cemrl_td3.cemrl import get_std_statistics
from evorl.algorithms.contrib.erl_origin import get_ec_pop_statistics
from evorl.algorithms.ec.ec_agent import DeterministicECAgent
from evorl.envs import Box
from evorl.sample_batch import SampleBatch
from evorl.networks.utils import model_state_axes
from evorl.utils.ec_utils import ParamVectorSpec

from tests.helpers.agents import build_agent


def test_flax_parameters_roundtrip_through_vector_spec():
    keys = jax.random.split(jax.random.key(0), 5)

    batch_params = [
        nnx.state(nnx.Linear(3, 7, rngs=nnx.Rngs(key)), nnx.Param) for key in keys
    ]
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


def test_parameter_statistics_are_plain_serializable_values():

    params = nnx.state(nnx.Linear(3, 2, rngs=nnx.Rngs(0)), nnx.Param)
    variance = jax.tree.map(lambda x: jnp.ones_like(x), params)
    for statistics in (
        _get_std_statistics(variance),
        get_std_statistics(variance),
        get_ec_pop_statistics(params),
    ):
        assert json.loads(json.dumps(statistics)) == statistics


def test_policy_graphstate_roundtrip_keeps_auxiliary_variables_out_of_genome(jit):
    """Evolutionary vectors contain weights; nested populations share SN state."""
    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = Box(low=-jnp.ones(2), high=jnp.ones(2))
    agent = build_agent(
        DeterministicECAgent,
        obs_space,
        action_space,
        actor_hidden_layer_sizes=(4,),
        norm_layer_type="spectral_norm",
    )
    initial = agent.init(obs_space, action_space, jax.random.key(0))
    models = nnx.merge(initial.model_state, copy=True)
    params = nnx.state(models.policy, nnx.Param)
    auxiliary = nnx.state(models.policy, nnx.Not(nnx.Param))
    assert jax.tree.leaves(auxiliary)
    assert all(isinstance(value, nnx.Param) for _, value in nnx.to_flat_state(params))
    spec = ParamVectorSpec(params)
    vectors = jnp.broadcast_to(spec.to_vector(params), (2, 3, spec.vec_size))
    genes = spec.to_tree(vectors)
    nnx.update(models.policy, genes)
    population = initial.replace(model_state=nnx.unpack(models))
    axes = initial.replace(
        obs_preprocessor_state=None,
        model_state=model_state_axes(
            population.model_state, nnx.StateAxes({nnx.Param: 0, ...: None})
        ),
    )

    def evaluate(member):
        return agent.evaluate_actions(
            member, SampleBatch(obs=jnp.ones((5, 3))), jax.random.key(1)
        )[0]

    evaluate_population = jax.vmap(jax.vmap(evaluate, in_axes=(axes,)), in_axes=(axes,))
    actions = (jax.jit(evaluate_population) if jit else evaluate_population)(population)
    expected = evaluate(initial)
    chex.assert_trees_all_close(
        actions, jnp.broadcast_to(expected, (2, 3, *expected.shape))
    )
    chex.assert_trees_all_equal(
        auxiliary,
        nnx.state(
            nnx.merge(population.model_state, copy=True).policy, nnx.Not(nnx.Param)
        ),
    )
