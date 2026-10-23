"""Scalar and vector critics preserve distinct output contracts under JIT."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.networks import (
    QNetwork,
    DiscreteQNetwork,
    MultiObjectiveQNetwork,
    MultiObjectiveDiscreteQNetwork,
    make_q_network,
    make_discrete_q_network,
    make_multi_objective_q_network,
    make_multi_objective_discrete_q_network,
)


@pytest.mark.parametrize("num_objectives", [None, 1, 3])
@pytest.mark.parametrize("num_critics", [1, 2])
@pytest.mark.parametrize("continuous", [False, True])
def test_vector_q_shapes_gradients_and_nested_batches(
    num_objectives, num_critics, continuous, jit
):
    kwargs = dict(
        input_size=3,
        action_size=2,
        hidden_layer_sizes=[4],
        n_stack=num_critics,
        rngs=nnx.Rngs(0),
    )
    if num_objectives is None:
        factory = make_q_network if continuous else make_discrete_q_network
        expected_type = QNetwork if continuous else DiscreteQNetwork
        network = factory(**kwargs)
    else:
        factory = (
            make_multi_objective_q_network
            if continuous
            else make_multi_objective_discrete_q_network
        )
        expected_type = (
            MultiObjectiveQNetwork if continuous else MultiObjectiveDiscreteQNetwork
        )
        network = factory(num_objectives=num_objectives, **kwargs)
    assert type(network) is expected_type
    assert isinstance(network, QNetwork if continuous else DiscreteQNetwork)
    state = nnx.unpack(network)
    observations = jnp.ones((2, 3, 4, 3))
    actions = jnp.ones((2, 3, 4, 2))

    def run(state, obs, actions):
        model = nnx.merge(state, copy=True)
        return model(obs, actions) if continuous else model(obs)

    outputs = (jax.jit(run) if jit else run)(state, observations, actions)
    expected = (2, 3, 4) + ((num_critics,) if num_critics > 1 else ())
    if not continuous:
        expected += (2,)
    if num_objectives is not None:
        expected += (num_objectives,)
    assert outputs.shape == expected
    nested = jax.jit(
        jax.vmap(jax.vmap(run, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))
    )(state, observations, actions)
    np.testing.assert_allclose(nested, outputs, atol=1e-6)
    grads = jax.grad(lambda state: run(state, observations, actions).sum())(state)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
