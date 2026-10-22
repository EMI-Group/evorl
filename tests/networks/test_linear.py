"""Native NNX networks under functional JAX transformations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from evorl.networks import (
    MLP,
    StaticLayerNorm,
    make_discrete_q_network,
    make_mlp,
    make_policy_network,
    make_q_network,
    make_v_network,
    make_vmap_mlp,
)
from evorl.networks.spectral_norm import SNDense, SingularVector


def test_mlp_matches_explicit_linear_layers_and_final_activation():
    model = MLP(
        2,
        (3, 1),
        activation=jax.nn.tanh,
        activation_final=jax.nn.sigmoid,
        rngs=nnx.Rngs(0),
    )
    model.layers[0].kernel[...] = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 10
    model.layers[0].bias[...] = jnp.array([0.1, -0.2, 0.3])
    model.layers[1].kernel[...] = jnp.array([[0.5], [-0.2], [0.1]])
    model.layers[1].bias[...] = jnp.array([0.4])
    x = jnp.array([[1.0, 2.0], [-1.0, 0.5]])
    expected = jax.nn.sigmoid(
        jnp.tanh(x @ model.layers[0].kernel[...] + model.layers[0].bias[...])
        @ model.layers[1].kernel[...]
        + model.layers[1].bias[...]
    )
    np.testing.assert_allclose(model(x), expected, rtol=1e-6)


def test_static_layer_norm_has_no_trainable_scale_or_bias():
    model = StaticLayerNorm(3, fixed_scale=2.0, fixed_bias=0.5, rngs=nnx.Rngs(0))
    x = jnp.array([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]])
    expected = (x - x.mean(-1, keepdims=True)) * jax.lax.rsqrt(
        x.var(-1, keepdims=True) + 1e-6
    ) * 2.0 + 0.5
    np.testing.assert_allclose(model(x), expected, rtol=1e-6)
    assert not jax.tree.leaves(nnx.state(model, nnx.Param))


@pytest.mark.parametrize(
    "norm", ["none", "layer_norm", "static_layer_norm", "spectral_norm"]
)
def test_abstract_construction_matches_real_graph_and_initialization_is_functional(
    norm,
):
    def create(key):
        return make_policy_network(
            3, 2, hidden_layer_sizes=(8,), norm_layer_type=norm, rngs=nnx.Rngs(key)
        )

    graph, abstract = nnx.split(nnx.eval_shape(create, jax.random.key(0)))
    assert all(isinstance(x, jax.ShapeDtypeStruct) for x in jax.tree.leaves(abstract))

    @jax.jit
    def initialize(key):
        model = create(key)
        real_graph, state = nnx.split(model)
        assert real_graph == graph
        return state

    state = initialize(jax.random.key(1))
    state2 = initialize(jax.random.key(2))
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(state2), strict=True)
    )
    batched = jax.jit(jax.vmap(jax.vmap(initialize)))(
        jax.random.split(jax.random.key(3), (2, 3))
    )

    def apply(state, x):
        return nnx.merge(graph, state, copy=True)(x)

    output = jax.jit(jax.vmap(jax.vmap(apply, in_axes=(0, None)), in_axes=(0, None)))(
        batched, jnp.ones((4, 3))
    )
    assert output.shape == (2, 3, 4, 2)
    assert jnp.all(jnp.isfinite(output))


@pytest.mark.parametrize(
    "norm", ["none", "layer_norm", "static_layer_norm", "spectral_norm"]
)
def test_ensemble_matches_separate_network_calls_and_gradients(norm):
    ensemble = make_vmap_mlp(3, (5, 2), 3, norm_layer_type=norm, rngs=nnx.Rngs(0))
    graph, params, other = nnx.split(ensemble, nnx.Param, ...)
    member_graph, member_state = nnx.split(ensemble.mlp)
    inputs = jnp.arange(36, dtype=jnp.float32).reshape(3, 4, 3) / 36

    expected = []
    expected_other = []
    for i in range(3):
        state = jax.tree.map(lambda x, index=i: x[index], member_state)
        member = nnx.merge(member_graph, state, copy=True)
        expected.append(member(inputs[i]))
        expected_other.append(nnx.state(member, nnx.Not(nnx.Param)))
    expected = jnp.stack(expected, axis=-2)

    @jax.jit
    def forward(params, other):
        model = nnx.merge(graph, params, other, copy=True)
        output = model(inputs)
        return jnp.mean(output**2), (output, nnx.state(model, nnx.Not(nnx.Param)))

    (loss, (output, updated)), grads = jax.value_and_grad(forward, has_aux=True)(
        params, other
    )
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)
    assert output.shape == (4, 3, 2)
    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads))
    assert all(x.shape[0] == 3 for x in jax.tree.leaves(params))
    kernels = params["mlp"]["layers"][0]["kernel"][...]
    assert not np.array_equal(kernels[0], kernels[1])
    if norm == "spectral_norm":
        expected_state = jax.tree.map(lambda *x: jnp.stack(x), *expected_other)
        for actual, desired in zip(
            jax.tree.leaves(updated["mlp"]),
            jax.tree.leaves(expected_state),
            strict=True,
        ):
            np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_stack", [1, 2, 3])
@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 4)])
def test_policy_value_and_critic_shapes(n_stack, batch_shape):
    obs = jnp.ones((*batch_shape, 3))
    actions = jnp.ones((*batch_shape, 2))
    kwargs = dict(hidden_layer_sizes=(8,), rngs=nnx.Rngs(0))
    policy = make_policy_network(3, 2, obs_key="state", **kwargs)
    value = make_v_network(3, obs_key="state", **kwargs)
    q = make_q_network(3, 2, n_stack=n_stack, obs_key="state", **kwargs)
    discrete_q = make_discrete_q_network(
        3, 4, n_stack=n_stack, obs_key="state", **kwargs
    )
    assert policy({"state": obs}).shape == (*batch_shape, 2)
    assert value({"state": obs}).shape == batch_shape
    expected_q = (*batch_shape, n_stack) if n_stack > 1 else batch_shape
    assert q({"state": obs}, actions).shape == expected_q
    expected_discrete = (*batch_shape, n_stack, 4) if n_stack > 1 else (*batch_shape, 4)
    assert discrete_q({"state": obs}).shape == expected_discrete


def test_spectral_normalization_matches_power_iteration_and_separates_state():
    model = SNDense(2, 2, n_steps=3, rngs=nnx.Rngs(0))
    kernel = jnp.array([[2.0, 1.0], [-0.5, 3.0]])
    u = jnp.array([[1.0, 2.0]])
    model.kernel[...] = kernel
    model.bias[...] = jnp.array([0.1, -0.2])
    model.u0[...] = u
    x = jnp.array([[1.0, 2.0]])
    for _ in range(3):
        v = u @ kernel.T
        v = v / jnp.sqrt(jnp.sum(v**2, keepdims=True) + model.eps)
        u = v @ kernel
        u = u / jnp.sqrt(jnp.sum(u**2, keepdims=True) + model.eps)
    sigma = (v @ kernel @ u.T)[0, 0]
    expected = x @ (kernel / sigma) + model.bias[...]
    np.testing.assert_allclose(model(x), expected, rtol=1e-6)
    np.testing.assert_allclose(model.u0[...], u, rtol=1e-6)
    params = nnx.state(model, nnx.Param)
    assert set(params) == {"kernel", "bias"}
    assert set(nnx.state(model, SingularVector)) == {"u0"}


def test_network_configuration_validation():
    with pytest.raises(ValueError, match="n_stack"):
        make_q_network(3, 2, n_stack=0, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="norm_layer_type"):
        make_mlp(3, (2,), norm_layer_type="unknown", rngs=nnx.Rngs(0))


def test_zero_spectral_kernel_has_finite_gradients_and_recovers_direction():
    model = SNDense(2, 2, rngs=nnx.Rngs(0))
    model.kernel[...] = jnp.zeros((2, 2))
    previous_direction = np.asarray(model.u0[...]).copy()
    graphdef, params, other = nnx.split(model, nnx.Param, ...)

    @jax.jit
    def differentiate(params, other):
        def loss(params):
            local = nnx.merge(graphdef, params, other, copy=True)
            output = local(jnp.ones((3, 2)))
            return output.sum(), (output, nnx.state(local, SingularVector))

        return jax.value_and_grad(loss, has_aux=True)(params)

    (loss, (output, updated)), grads = differentiate(params, other)
    assert jnp.isfinite(loss)
    np.testing.assert_array_equal(output, jnp.zeros((3, 2)))
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads))
    np.testing.assert_array_equal(updated["u0"][...], previous_direction)
    # A nonzero update must resume ordinary power iteration instead of keeping
    # a collapsed zero vector forever.
    recovered = nnx.merge(graphdef, params, updated, copy=True)
    recovered.kernel[...] = jnp.array([[2.0, 1.0], [-0.5, 3.0]])
    recovered(jnp.ones((3, 2)))
    assert jnp.all(jnp.isfinite(recovered.u0[...]))
    assert not np.array_equal(recovered.u0[...], previous_direction)
