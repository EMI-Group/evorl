"""Decomposition, PSA and maximization archive formulas for MORL/D."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.algorithms.multi_objective.morld_utils import (
    decomposition_weights,
    nearest_decomposition_neighbors,
    psa_preferences,
    update_decomposition_archive,
)
from evorl.types import PyTreeDict


@pytest.mark.parametrize("objectives", [1, 2, 3, 4])
def test_decomposition_directions_are_normalized_and_distinct(objectives):
    weights = decomposition_weights(objectives, 6)
    assert weights.shape == (6, objectives)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(-1), 1)
    if objectives > 1:
        assert len(np.unique(weights, axis=0)) == 6


def test_neighbors_exclude_self_and_psa_uses_nearest_distinct_front_point():
    weights = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    np.testing.assert_array_equal(
        nearest_decomposition_neighbors(weights, 1), [[1], [0], [1]]
    )
    updated = jax.jit(psa_preferences)(
        weights[1:2],
        jnp.array([[3.0, 1.0]]),
        jnp.array([[3.0, 1.0], [1.0, 3.0]]),
        jnp.array([True, True]),
        1.0,
    )
    np.testing.assert_allclose(updated, [[0.8, 0.2]])
    unchanged = psa_preferences(weights, weights, weights, jnp.zeros(3, bool), 0.1)
    np.testing.assert_array_equal(unchanged, weights)


def test_archive_rejects_dominated_duplicates_and_preserves_matching_snapshots():
    archive = PyTreeDict(
        returns=jnp.zeros((3, 2)),
        valid=jnp.zeros(3, bool),
        preferences=jnp.zeros((3, 2)),
        agent_states=PyTreeDict(identity=jnp.zeros(3)),
    )
    values = jnp.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    updated = jax.jit(update_decomposition_archive)(
        archive,
        values,
        PyTreeDict(identity=jnp.arange(5.0)),
        jnp.ones((5, 2)) / 2,
    )
    assert bool(updated.valid.all())
    assert set(np.asarray(updated.agent_states.identity).tolist()) == {0, 1, 2}
    np.testing.assert_array_equal(
        values[updated.agent_states.identity.astype(int)], updated.returns
    )
    improved = update_decomposition_archive(
        updated,
        jnp.array([[4.0, 4.0]]),
        PyTreeDict(identity=jnp.array([9.0])),
        jnp.array([[0.5, 0.5]]),
    )
    assert int(improved.valid.sum()) == 1
    np.testing.assert_array_equal(improved.returns[improved.valid], [[4.0, 4.0]])


def test_archive_capacity_keeps_extreme_policies():
    archive = PyTreeDict(
        returns=jnp.zeros((2, 2)),
        valid=jnp.zeros(2, bool),
        preferences=jnp.zeros((2, 2)),
        agent_states=jnp.zeros(2),
    )
    updated = update_decomposition_archive(
        archive,
        jnp.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]),
        jnp.arange(3.0),
        jnp.ones((3, 2)) / 2,
    )
    assert set(np.asarray(updated.agent_states).tolist()) == {0, 2}
