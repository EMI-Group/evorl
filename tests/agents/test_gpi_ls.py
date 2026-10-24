"""Independent GPI maximization, DDQN-vector and linear-support regressions."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf
import pytest

from evorl.algorithms.multi_objective.gpi_ls import (
    GPILSAgent,
    convex_coverage_indices,
    gpi_actions,
    linear_support_corners,
)
from evorl.envs import Box, Discrete
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict


def test_gpi_scores_all_support_policies_using_the_query_weight():
    # Policy 0's own utility favors action 0. Policy 1 supplies action 1's
    # superior first coordinate, so GPI must select it for query [1, 0].
    values = jnp.array([[[[2.0, 100.0], [1.0, 0.0]], [[0.0, 100.0], [5.0, 0.0]]]])
    query = jnp.array([[1.0, 0.0]])
    np.testing.assert_array_equal(
        gpi_actions(values, query, jnp.array([True, True])), [1]
    )
    np.testing.assert_array_equal(
        gpi_actions(values, query, jnp.array([True, False])), [0]
    )
    nested = jax.jit(jax.vmap(jax.vmap(gpi_actions)))
    actual = nested(values[None, None], query[None, None], jnp.ones((1, 1, 2), bool))
    np.testing.assert_array_equal(actual, [[[1]]])


@pytest.mark.parametrize("objectives", [1, 2, 3])
def test_exact_corners_include_simplex_boundaries_and_crossing_planes(objectives):
    values = np.eye(objectives) * 3
    corners = linear_support_corners(values, objectives)
    np.testing.assert_allclose(corners.sum(-1), 1)
    assert (corners >= 0).all()
    for extreme in np.eye(objectives):
        assert any(np.allclose(extreme, corner) for corner in corners)
    assert any(
        np.allclose(np.full(objectives, 1 / objectives), corner) for corner in corners
    )


def test_convex_coverage_removes_dominated_and_unsupported_vectors():
    values = np.array([[3, 0], [0, 3], [2, 2], [1, 1], [2, 2], [-1, -1]])
    np.testing.assert_array_equal(convex_coverage_indices(values), [0, 1, 2])
    corners = linear_support_corners(values[:3], 2)
    assert any(np.allclose(weight, [1 / 3, 2 / 3]) for weight in corners)
    assert any(np.allclose(weight, [2 / 3, 1 / 3]) for weight in corners)


def test_convex_coverage_discards_dominated_boundary_ties():
    values = np.array([[3, 0], [0, 3], [3, -1], [3, -2], [3, -3]])
    np.testing.assert_array_equal(convex_coverage_indices(values), [0, 1])
    # A distinct nondominated point tied at the balanced preference remains
    # useful and is retained with the existing tie policy.
    np.testing.assert_array_equal(
        convex_coverage_indices(np.array([[3, 0], [0, 3], [1.5, 1.5]])), [0, 1, 2]
    )


class ConstantQ(nnx.Module):
    def __init__(self, values):
        self.values = nnx.Param(jnp.asarray(values, jnp.float32))

    def __call__(self, inputs):
        return jnp.broadcast_to(self.values[...], (*inputs.shape[:-1], 2, 2))


def test_ddqn_bootstrap_preserves_one_action_vector_and_masks_only_termination():
    agent = GPILSAgent(
        num_objectives=2,
        num_actions=2,
        max_support_size=2,
        discount=0.5,
        agent_network_cfg=OmegaConf.create(dict(q_hidden_layer_sizes=[4])),
    )
    state = agent.init(
        Box(low=-jnp.ones(1), high=jnp.ones(1)), Discrete(n=2), jax.random.key(0)
    )
    state = state.replace(
        model_state=nnx.unpack(
            nnx.Dict(
                q=ConstantQ([[2, 0], [0, 3]]),
                target_q=ConstantQ([[10, 20], [30, 40]]),
            )
        ),
        extra_state=state.extra_state.replace(
            training_preference=jnp.array([1.0, 0.0]),
            support_weights=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
            support_mask=jnp.array([True, False]),
        ),
    )
    batch = SampleBatch(
        obs=jnp.zeros((2, 1)),
        actions=jnp.zeros(2, jnp.int32),
        rewards=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=jnp.zeros((2, 1)), termination=jnp.array([0.0, 1.0])
            )
        ),
    )
    info = jax.jit(agent.loss)(state, batch, jax.random.key(1))
    # Query e0 chooses target action 0; support e1 chooses action 1.
    targets = jnp.array([[[6.0, 12.0], [16.0, 22.0]], [[3.0, 4.0], [3.0, 4.0]]])
    expected = optax.huber_loss(
        jnp.broadcast_to(jnp.array([2.0, 0.0]), targets.shape), targets
    ).mean()
    np.testing.assert_allclose(info.q_loss, expected)
    assert np.isfinite(info.q_loss)
