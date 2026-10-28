"""Native evolutionary optimizers, selection semantics and generation updates."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.ec.operators.selection.non_dominated import non_dominated_sort
from evorl.ec.optimizers.nsga2 import crowding_distance
from evorl.ec.optimizers import CMAES, NSGA2, SepCMAES
from evorl.utils.ec_utils import ParamVectorSpec

from tests.helpers.sharding import compiled_collectives


def test_crowding_distance_normalizes_each_front_and_ignores_constant_objectives():
    fitnesses = jnp.array([[0, 3], [1, 2], [3, 0], [-1, -1], [-1, -1]], jnp.float32)
    ranks = non_dominated_sort(fitnesses)
    np.testing.assert_array_equal(ranks, [0, 0, 0, 1, 1])
    distance = jax.jit(crowding_distance)(fitnesses, ranks)
    np.testing.assert_array_equal(distance, [jnp.inf, 2, jnp.inf, 0, 0])
    extended = jnp.concatenate((fitnesses, jnp.zeros((5, 1))), axis=1)
    np.testing.assert_array_equal(crowding_distance(extended, ranks), distance)


@pytest.mark.parametrize("num_objectives", [0, 1])
def test_nsga2_rejects_fewer_than_two_objectives(num_objectives):
    with pytest.raises(ValueError, match="at least two"):
        NSGA2(
            pop_size=4,
            num_objectives=num_objectives,
            param_vec_spec=ParamVectorSpec(jnp.zeros(2)),
        )


@pytest.mark.parametrize("optimizer_cls", [CMAES, SepCMAES])
def test_cmaes_update_matches_reference_equations(optimizer_cls):
    mean = {"w": jnp.array([[1.0, 2.0]])}
    optimizer = optimizer_cls(
        pop_size=4, num_elites=2, init_stdev=0.5, param_vec_spec=ParamVectorSpec(mean)
    )
    initial = optimizer.init(mean, jax.random.key(0))
    population = np.array([[1.5, 2], [0, 3], [1, 1], [2, 2.5]], dtype=np.float32)
    fitnesses = jnp.array([2, 0, 1, 3], jnp.float32)
    pending = initial.replace(
        population=jnp.asarray(population), count_iter=jnp.int32(1)
    )
    _, actual = jax.jit(optimizer.tell)(pending, fitnesses=fitnesses)
    weights = np.asarray(optimizer.weights)
    y = (population[[3, 0]] - [1, 2]) / 0.5
    yw = weights @ y
    ps = np.sqrt(optimizer.cs * (2 - optimizer.cs) * optimizer.mueff) * yw
    hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - optimizer.cs) ** 2)
    hsig = float(hsig < (1.4 + 2 / 3) * float(optimizer.chiN))
    pc = hsig * np.sqrt(optimizer.cc * (2 - optimizer.cc) * optimizer.mueff) * yw
    factor = 1 + optimizer.c1 * (1 - hsig) * optimizer.cc * (2 - optimizer.cc)
    factor -= optimizer.c1 + optimizer.cmu
    if optimizer_cls is CMAES:
        expected_c = factor * np.eye(2) + optimizer.c1 * np.outer(pc, pc)
        expected_c += optimizer.cmu * (y.T * weights) @ y
    else:
        expected_c = factor + optimizer.c1 * pc**2 + optimizer.cmu * (weights @ y**2)
    expected_sigma = 0.5 * np.exp(
        optimizer.cs
        / optimizer.damps
        * (np.linalg.norm(ps) / float(optimizer.chiN) - 1)
    )
    np.testing.assert_allclose(actual.mean, [1, 2] + 0.5 * yw, rtol=1e-6)
    np.testing.assert_allclose(actual.ps, ps, rtol=1e-6)
    np.testing.assert_allclose(actual.pc, pc, atol=1e-6)
    np.testing.assert_allclose(actual.C, expected_c, rtol=1e-6)
    np.testing.assert_allclose(actual.sigma, expected_sigma, rtol=1e-6)


@pytest.mark.parametrize("optimizer_cls", [CMAES, SepCMAES])
@pytest.mark.parametrize("pop_size,num_elites", [(4, 3), (4, 4), (5, 3)])
def test_cmaes_rejects_elites_that_would_receive_nonpositive_weights(
    optimizer_cls, pop_size, num_elites
):
    with pytest.raises(ValueError, match="pop_size // 2"):
        optimizer_cls(
            pop_size=pop_size,
            num_elites=num_elites,
            init_stdev=1.0,
            param_vec_spec=ParamVectorSpec(jnp.zeros(2)),
        )


@pytest.mark.parametrize("optimizer_cls", [CMAES, SepCMAES])
@pytest.mark.parametrize("pop_size,num_elites", [(4, 2), (5, 2), (6, 2)])
def test_cmaes_positive_weights_and_rates_follow_the_tutorial(
    optimizer_cls, pop_size, num_elites
):
    optimizer = optimizer_cls(
        pop_size=pop_size,
        num_elites=num_elites,
        init_stdev=1.0,
        param_vec_spec=ParamVectorSpec(jnp.zeros(3)),
    )
    weights = np.log((pop_size + 1) / 2) - np.log(np.arange(1, num_elites + 1))
    weights /= weights.sum()
    mu_eff = 1 / np.sum(weights**2)
    np.testing.assert_allclose(optimizer.weights, weights, rtol=1e-6)
    assert np.all(np.asarray(optimizer.weights) > 0)
    np.testing.assert_allclose(optimizer.mueff, mu_eff, rtol=1e-12)
    # Appendix C, lines 29-32. In particular c_c uses 4 + mu_eff / n.
    np.testing.assert_allclose(
        optimizer.cc, (4 + mu_eff / 3) / (3 + 4 + 2 * mu_eff / 3)
    )
    np.testing.assert_allclose(optimizer.c1, 2 / ((3 + 1.3) ** 2 + mu_eff))
    np.testing.assert_allclose(
        optimizer.cmu,
        min(1 - optimizer.c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((3 + 2) ** 2 + mu_eff)),
    )


def test_cmaes_whitening_matches_inverse_square_root_of_rotated_covariance():
    optimizer = CMAES(
        pop_size=4,
        num_elites=2,
        init_stdev=1.0,
        param_vec_spec=ParamVectorSpec(jnp.zeros(3)),
    )
    angle = np.pi / 6
    basis = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    scales = np.array([0.2, 1.7, 3.0])
    covariance = (basis * scales**2) @ basis.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    ps, update = np.array([0.2, -0.3, 0.1]), np.array([0.5, -1, 2])
    expected = (1 - optimizer.cs) * ps + np.sqrt(
        optimizer.cs * (2 - optimizer.cs) * optimizer.mueff
    ) * (inverse_sqrt @ update)
    actual = jax.jit(optimizer._update_ps)(
        *(jnp.asarray(x, jnp.float32) for x in (ps, basis, scales, update))
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("optimizer_cls", [CMAES, SepCMAES])
def test_cmaes_converges_with_matrix_parameters(optimizer_cls):
    mean = {"w": jnp.full((2, 2), 3.0)}
    optimizer = optimizer_cls(
        pop_size=16, num_elites=8, init_stdev=1.0, param_vec_spec=ParamVectorSpec(mean)
    )
    state = optimizer.init(mean, jax.random.key(3))

    def generation(state, _):
        population, state = optimizer.ask(state)
        fitnesses = -jnp.square(population["w"]).sum(axis=(1, 2))
        _, state = optimizer.tell(state, fitnesses)
        return state, None

    state, _ = jax.jit(lambda s: jax.lax.scan(generation, s, None, length=100))(state)
    assert np.linalg.norm(state.mean) < 0.01
    assert int(state.count_iter) == 100
    assert np.isfinite(state.sigma) and state.sigma > 0
    eigenvalues = np.linalg.eigvalsh(state.C) if optimizer_cls is CMAES else state.C
    assert np.all(eigenvalues > 0)


def test_nsga2_preserves_elite_parents_and_pareto_extremes():
    optimizer = NSGA2(
        pop_size=4, num_objectives=2, param_vec_spec=ParamVectorSpec(jnp.zeros(2))
    )
    state = optimizer.init(jax.random.key(1))
    parents = jnp.array([[0, 3], [1, 2], [2, 1], [3, 0]], jnp.float32)
    state = state.replace(offspring=parents)
    _, state = jax.jit(optimizer.tell)(state, parents)
    state = state.replace(offspring=jnp.full((4, 2), -5.0))
    _, survived = jax.jit(optimizer.tell)(state, state.offspring)
    assert set(map(tuple, np.asarray(survived.population))) == set(
        map(tuple, np.asarray(parents))
    )
    offspring = jnp.array([[0.5, 2.5], [1.5, 1.5], [2.5, 0.5], [1.7, 1.3]])
    _, state = optimizer.tell(state.replace(offspring=offspring), offspring)
    assert np.any(np.all(np.asarray(state.population) == [0, 3], axis=-1))
    assert np.any(np.all(np.asarray(state.population) == [3, 0], axis=-1))
    np.testing.assert_array_equal(state.ranks, np.zeros(4))


@pytest.mark.parametrize("pop_size", [7, 8])
def test_nsga2_optimizes_zdt1_with_bounded_offspring(pop_size):
    optimizer = NSGA2(
        pop_size=pop_size,
        num_objectives=2,
        param_vec_spec=ParamVectorSpec({"w": jnp.zeros((1, 5))}),
        lb=0,
        ub=1,
    )
    state = optimizer.init(jax.random.key(42))

    def generation(state):
        population, state = optimizer.ask(state)
        x = population["w"][:, 0]
        g = 1 + 9 * x[:, 1:].mean(axis=-1)
        fitnesses = -jnp.stack((x[:, 0], g * (1 - jnp.sqrt(x[:, 0] / g))), axis=-1)
        return optimizer.tell(state, fitnesses)[1]

    step = jax.jit(generation)
    initial = step(state)
    state = initial
    for _ in range(100):
        state = step(state)
        assert np.all(np.isfinite(state.population))
        assert np.all((state.population >= 0) & (state.population <= 1))
    assert (
        np.asarray(state.population[:, 1:]).mean()
        < np.asarray(initial.population[:, 1:]).mean() / 2
    )
    assert np.max(np.asarray(state.population[:, 1:]).mean(axis=-1)) < 0.15


@pytest.mark.parametrize("optimizer_cls", [CMAES, SepCMAES, NSGA2])
def test_native_optimizers_support_nested_vmap(optimizer_cls):
    mean = {"w": jnp.ones((2, 2))}
    spec = ParamVectorSpec(mean)
    if optimizer_cls is NSGA2:
        optimizer = optimizer_cls(pop_size=4, num_objectives=2, param_vec_spec=spec)
        initialize = optimizer.init
    else:
        optimizer = optimizer_cls(
            pop_size=4, num_elites=2, init_stdev=0.1, param_vec_spec=spec
        )
        initialize = lambda key: optimizer.init(mean, key)

    def run(key):
        state = initialize(key)
        # Full CMA-ES eigenvectors can change sign between batched and serial
        # decompositions, changing later samples without changing their law.
        # Compare the first update; convergence is checked separately above.
        for _ in range(1 if optimizer_cls is CMAES else 2):
            population, state = optimizer.ask(state)
            fitnesses = -jnp.square(population["w"]).sum(axis=(1, 2))
            if optimizer_cls is NSGA2:
                fitnesses = jnp.stack((fitnesses, -fitnesses), axis=-1)
            _, state = optimizer.tell(state, fitnesses)
        return state

    keys = jax.random.split(jax.random.key(9), (2, 2))
    batched = jax.jit(jax.vmap(jax.vmap(run)))(keys)
    for i in range(2):
        for j in range(2):
            actual = jax.tree.map(lambda x, i=i, j=j: x[i, j], batched)
            expected = jax.jit(run)(keys[i, j])
            chex.assert_trees_all_equal(
                jax.random.key_data(actual.key), jax.random.key_data(expected.key)
            )
            chex.assert_trees_all_close(
                actual.replace(key=None),
                expected.replace(key=None),
                atol=2e-5,
                rtol=2e-5,
            )


@pytest.mark.parametrize("optimizer_cls", [CMAES, SepCMAES, NSGA2])
def test_replicated_optimizers_need_no_communication(optimizer_cls, mesh_factory):
    mean = jnp.ones(3)
    spec = ParamVectorSpec(mean)
    if optimizer_cls is NSGA2:
        optimizer = optimizer_cls(pop_size=4, num_objectives=2, param_vec_spec=spec)
        state = optimizer.init(jax.random.key(7))
    else:
        optimizer = optimizer_cls(
            pop_size=4, num_elites=2, init_stdev=0.1, param_vec_spec=spec
        )
        state = optimizer.init(mean, jax.random.key(7))

    def generation(state):
        population, state = optimizer.ask(state)
        fitnesses = -jnp.square(population).sum(axis=-1)
        if optimizer_cls is NSGA2:
            fitnesses = jnp.stack((fitnesses, -fitnesses), axis=-1)
        return optimizer.tell(state, fitnesses)[1]

    mesh = mesh_factory(shape=(min(2, jax.local_device_count()),))
    with jax.set_mesh(mesh):
        state = jax.device_put(state, jax.P())
        mapped = jax.jit(jax.shard_map(generation, in_specs=jax.P(), out_specs=jax.P()))
        executable = mapped.lower(state).compile()
        state = executable(state)
        # Exercise NSGA-II's post-initialization offspring/survival branches too.
        jax.block_until_ready(executable(state))
    assert not compiled_collectives(executable.as_text())
