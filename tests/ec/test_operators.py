"""MLP variation preserves parameter structure, parent values and LayerNorm."""

from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.ec.operators import MLPCrossover, MLPMutation, mlp_crossover, mlp_mutate


@pytest.fixture
def parameters():
    return {
        "params": {
            "Dense_0": {
                "kernel": jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4) / 100,
                "bias": jnp.arange(1, 5, dtype=jnp.float32) / 100,
            },
            "LayerNorm_0": {"scale": jnp.full(4, 2.0), "bias": jnp.full(4, 3.0)},
        }
    }


def test_zero_mutation_preserves_all_parameters(parameters, jit):
    mutate = partial(mlp_mutate, matrix_num_mutation_frac=0, vector_num_mutation_frac=0)
    actual = (jax.jit(mutate) if jit else mutate)(parameters, jax.random.key(1))
    chex.assert_trees_all_equal(actual, parameters)


def test_mutation_changes_requested_kernel_entries_and_preserves_bias_and_norm(
    parameters, jit
):
    mutate = partial(mlp_mutate, matrix_num_mutation_frac=0.25, mut_strength=0.3)
    mutate = jax.jit(mutate) if jit else mutate
    key = jax.random.key(8)
    snapshot = jax.tree.map(lambda x: np.array(x, copy=True), parameters)
    actual = mutate(parameters, key)
    chex.assert_trees_all_equal(parameters, snapshot)
    chex.assert_trees_all_equal(actual, mutate(parameters, key))
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, parameters)
    before, after = parameters["params"], actual["params"]
    assert (
        np.count_nonzero(
            np.asarray(after["Dense_0"]["kernel"] - before["Dense_0"]["kernel"])
        )
        == 4
    )
    np.testing.assert_array_equal(after["Dense_0"]["bias"], before["Dense_0"]["bias"])
    chex.assert_trees_all_equal(after["LayerNorm_0"], before["LayerNorm_0"])


def test_full_mutation_clips_dense_parameters_but_not_layer_norm(parameters, jit):
    mutate = partial(
        mlp_mutate,
        matrix_num_mutation_frac=1,
        vector_num_mutation_frac=1,
        mut_strength=3,
        weight_max_magnitude=0.5,
    )
    actual = (jax.jit(mutate) if jit else mutate)(parameters, jax.random.key(9))
    for leaf in jax.tree.leaves(actual["params"]["Dense_0"]):
        assert np.all(np.isfinite(leaf))
        assert np.max(np.abs(leaf)) <= 0.5
    chex.assert_trees_all_equal(
        actual["params"]["LayerNorm_0"], parameters["params"]["LayerNorm_0"]
    )


@pytest.mark.parametrize("fraction", [0, 0.5, 1])
def test_crossover_copies_selected_parent_rows_without_creating_values(
    parameters, fraction, jit
):
    other = jax.tree.map(lambda x: -x, parameters)
    crossover = partial(mlp_crossover, num_crossover_frac=fraction)
    crossover = jax.jit(crossover) if jit else crossover
    children = crossover(parameters, other, jax.random.key(3))
    chex.assert_trees_all_equal(
        children, crossover(parameters, other, jax.random.key(3))
    )
    for parent, child in zip((parameters, other), children, strict=True):
        chex.assert_trees_all_equal_shapes_and_dtypes(parent, child)
        chex.assert_trees_all_equal(
            parent["params"]["LayerNorm_0"], child["params"]["LayerNorm_0"]
        )
    for name in ("kernel", "bias"):
        first = np.asarray(parameters["params"]["Dense_0"][name]).reshape(4, -1)
        second = np.asarray(other["params"]["Dense_0"][name]).reshape(4, -1)
        left, right = [
            np.asarray(x["params"]["Dense_0"][name]).reshape(4, -1) for x in children
        ]
        for child in (left, right):
            assert np.all(
                np.all(child == first, axis=-1) | np.all(child == second, axis=-1)
            )
        copied_rows = np.all(left == right, axis=-1)
        assert copied_rows.sum() == round(4 * fraction)
        np.testing.assert_array_equal(left[~copied_rows], first[~copied_rows])
        np.testing.assert_array_equal(right[~copied_rows], second[~copied_rows])


@pytest.mark.parametrize("operator", ["mutation", "crossover"])
@pytest.mark.parametrize("batched_key", [False, True])
def test_population_operators_match_independent_members(
    parameters, operator, batched_key, jit
):
    population = jax.tree.map(
        lambda x: jnp.stack([x + i / 10 for i in range(4)]), parameters
    )
    key = jax.random.key(11)
    count = 4 if operator == "mutation" else 2
    keys = jax.random.split(key, count)
    if operator == "mutation":
        variation = MLPMutation(matrix_num_mutation_frac=0.5)
        expected = [
            mlp_mutate(
                jax.tree.map(lambda x, i=i: x[i], population),
                keys[i],
                matrix_num_mutation_frac=0.5,
            )
            for i in range(4)
        ]
    else:
        variation = MLPCrossover(num_crossover_frac=0.5)
        pairs = [
            mlp_crossover(
                jax.tree.map(lambda x, i=i: x[2 * i], population),
                jax.tree.map(lambda x, i=i: x[2 * i + 1], population),
                keys[i],
                num_crossover_frac=0.5,
            )
            for i in range(2)
        ]
        expected = [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
    expected = jax.tree.map(lambda *xs: jnp.stack(xs), *expected)
    actual = (jax.jit(variation) if jit else variation)(
        population, keys if batched_key else key
    )
    chex.assert_trees_all_close(actual, expected, atol=1e-7)


def test_crossover_rejects_odd_populations(parameters):
    population = jax.tree.map(lambda x: jnp.stack([x, x, x]), parameters)
    with pytest.raises(AssertionError, match="pop_size must be even"):
        MLPCrossover()(population, jax.random.key(2))
