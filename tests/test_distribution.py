import jax
import jax.numpy as jnp
import chex
import numpy as np
import pytest
from scipy import special, stats
from evorl.distribution import (
    TanhTransformedDistribution,
    _normal_log_cdf,
    get_tanh_norm_dist,
    tfd,
)
from tests.helpers.sharding import compiled_collectives


def test_tanh_normal():
    T = 11
    B = 7
    A = 3

    loc = jnp.zeros((T, B, A))
    scale = jnp.ones((T, B, A))

    actions_dist = get_tanh_norm_dist(loc, scale)

    actions = jax.random.uniform(
        jax.random.key(42), shape=(T, B, A), minval=-0.999, maxval=0.999
    )

    logp = actions_dist.log_prob(actions)

    chex.assert_shape(logp, (T, B))


def test_tanh_normal_grad():
    T = 32
    B = 8
    A = 7

    loc = jnp.zeros((T, B, A))
    scale = jnp.ones((T, B, A))

    actions = jax.random.uniform(
        jax.random.key(42), shape=(T, B, A), minval=-1.0, maxval=1.0
    )

    def loss_fn(loc, scale):
        actions_dist = get_tanh_norm_dist(loc, scale)
        logp = actions_dist.log_prob(actions)

        return -logp.mean()

    loss, (g_loc, g_scale) = jax.value_and_grad(loss_fn, argnums=(0, 1))(loc, scale)

    assert not jnp.isnan(g_loc).any(), "loc grad has nan"
    assert not jnp.isnan(g_scale).any(), "scale grad has nan"


@pytest.mark.parametrize("action", [-1.0, -0.999, -0.2, 0.2, 0.999, 1.0])
def test_tanh_normal_boundary_probabilities_and_parameter_gradients(action):
    loc = jnp.array([-0.8, 0.6])
    raw_scale = jnp.array([-1.0, 0.5])
    actions = jnp.full(2, action)

    def reference(loc, raw_scale):
        scale = np.logaddexp(0, raw_scale) + 1e-3
        threshold = np.float32(0.999)
        inverse_threshold = np.arctanh(threshold).astype(np.float64)
        log_epsilon = np.log(1.0 - 0.999)
        if action <= -threshold:
            log_prob = special.log_ndtr((-inverse_threshold - loc) / scale)
            return np.sum(log_prob - log_epsilon)
        if action >= threshold:
            log_prob = special.log_ndtr((loc - inverse_threshold) / scale)
            return np.sum(log_prob - log_epsilon)
        inverse = np.arctanh(action)
        return np.sum(stats.norm.logpdf(inverse, loc, scale) - np.log1p(-(action**2)))

    def log_prob(loc, raw_scale):
        return get_tanh_norm_dist(loc, raw_scale).log_prob(actions)

    value, grads = jax.jit(jax.value_and_grad(log_prob, argnums=(0, 1)))(loc, raw_scale)
    params = [
        np.asarray(loc, dtype=np.float64),
        np.asarray(raw_scale, dtype=np.float64),
    ]
    np.testing.assert_allclose(value, reference(*params), rtol=2e-5, atol=2e-5)
    for group, actual in enumerate(grads):
        expected = np.zeros(2)
        for index in range(2):
            positive, negative = [x.copy() for x in params], [x.copy() for x in params]
            positive[group][index] += 1e-4
            negative[group][index] -= 1e-4
            expected[index] = (reference(*positive) - reference(*negative)) / 2e-4
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-5)


def normal_cdf_reference(x):
    """Use float64 SciPy and a stable Mills ratio as independent references."""
    x = np.asarray(x, dtype=np.float64)
    values = special.log_ndtr(x)
    derivative = np.empty_like(x)
    negative = x < 0
    derivative[negative] = np.sqrt(2 / np.pi) / special.erfcx(-x[negative] / np.sqrt(2))
    derivative[~negative] = np.exp(stats.norm.logpdf(x[~negative]) - values[~negative])
    return values, derivative


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["fp32", "fp64"])
def test_normal_log_cdf_forward_and_backward(dtype, jit):
    with jax.enable_x64(dtype == jnp.float64):
        cutoff = -20.0 if dtype == jnp.float64 else -10.0
        x = jnp.asarray(
            np.concatenate(
                (
                    -np.geomspace(10.01, 1e6, 800),
                    np.linspace(-20.1, 10, 2000),
                    [cutoff - 1e-4, cutoff, cutoff + 1e-4, -3800.2075, 0, 100],
                )
            ),
            dtype=dtype,
        )

        def loss(x):
            values = _normal_log_cdf(x)
            return values.sum(), values

        evaluate = jax.value_and_grad(loss, has_aux=True)
        if jit:
            evaluate = jax.jit(evaluate)
        (_, values), derivative = evaluate(x)
        expected_values, expected_derivative = normal_cdf_reference(x)
        assert values.dtype == derivative.dtype == dtype
        if dtype == jnp.float32:
            np.testing.assert_allclose(values, expected_values, rtol=3e-7, atol=2e-6)
            np.testing.assert_allclose(
                derivative, expected_derivative, rtol=1e-5, atol=1e-30
            )
        else:
            np.testing.assert_allclose(values, expected_values, rtol=3e-11, atol=5e-10)
            np.testing.assert_allclose(
                derivative, expected_derivative, rtol=3e-10, atol=1e-30
            )


def test_tanh_normal_forward_and_backward_grid(jit):
    loc, scale, action = (
        jnp.asarray(x.reshape(-1, 1), dtype=jnp.float32)
        for x in np.meshgrid(
            [-20, -10, -6, -3, -1, 0, 1, 3, 6, 10, 20],
            [0.001, 0.003, 0.01, 0.1, 0.3, 1, 3, 10],
            [
                -1,
                -0.9999,
                -0.999,
                -0.99899,
                -0.99,
                -0.5,
                0,
                0.5,
                0.99,
                0.99899,
                0.999,
                0.9999,
                1,
            ],
            indexing="ij",
        )
    )

    def loss(loc, scale, action):
        values = TanhTransformedDistribution(tfd.Normal(loc, scale)).log_prob(action)
        return values.sum(), values

    evaluate = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)
    if jit:
        evaluate = jax.jit(evaluate)
    (_, values), (dloc, dscale, daction) = evaluate(loc, scale, action)
    mu, sigma, event = (np.asarray(x, dtype=np.float64) for x in (loc, scale, action))
    threshold = float(np.float32(0.999))
    inverse_threshold = float(np.arctanh(np.float32(0.999)))
    left, right = event <= -threshold, event >= threshold
    interior = ~(left | right)
    clipped = np.clip(event, -threshold, threshold)
    z = np.where(
        left,
        (-inverse_threshold - mu) / sigma,
        np.where(
            right, (mu - inverse_threshold) / sigma, (np.arctanh(clipped) - mu) / sigma
        ),
    )
    cdf, mills = normal_cdf_reference(z)
    expected_values = np.where(
        interior,
        stats.norm.logpdf(z) - np.log(sigma) - np.log1p(-(clipped**2)),
        cdf - np.log(1.0 - 0.999),
    )
    expected_dloc = np.where(interior, z / sigma, np.where(left, -mills, mills) / sigma)
    expected_dscale = np.where(interior, (z**2 - 1) / sigma, -z * mills / sigma)
    expected_daction = np.where(
        interior,
        (-z / sigma + 2 * clipped) / (1 - clipped**2),
        0,
    )
    np.testing.assert_allclose(values, expected_values, rtol=3e-6, atol=3e-6)
    for actual, expected in (
        (dloc, expected_dloc),
        (dscale, expected_dscale),
        (daction, expected_daction),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.skipif(
    jax.__version_info__ == (0, 12, 0), reason="Other releases retain the TFP path"
)
def test_other_jax_versions_preserve_tfp_tail_probabilities():
    normal = tfd.Normal(jnp.array([-20.0, 0.0, 20.0]), jnp.array([0.001, 0.3, 1.0]))
    dist = TanhTransformedDistribution(normal)
    inverse = dist.bijector.inverse(0.999)
    epsilon = jnp.log(1.0 - 0.999)
    np.testing.assert_array_equal(
        dist._log_prob_left, normal.log_cdf(-inverse) - epsilon
    )
    np.testing.assert_array_equal(
        dist._log_prob_right, normal.log_survival_function(inverse) - epsilon
    )


def test_tanh_normal_sharded_gradients_need_no_communication(mesh_factory):
    loc = jnp.array([[-1.0, 0.5], [0.3, -0.6], [0.8, 0.1], [-0.4, 1.1]])
    loc = jnp.broadcast_to(loc[:, None, :], (4, 2, 2))
    raw_scale = jnp.broadcast_to(jnp.array([-100.0, -0.5]), loc.shape)
    actions = jnp.broadcast_to(
        jnp.array([-1.0, -0.999, -0.2, 0.2, 0.999, 1.0])[None, None, :, None],
        (4, 2, 6, 2),
    )

    def log_prob(loc, raw_scale, actions):
        return get_tanh_norm_dist(loc, raw_scale).log_prob(actions).sum()

    per_member = jax.vmap(jax.vmap(jax.value_and_grad(log_prob, argnums=(0, 1))))
    expected = jax.jit(per_member)(loc, raw_scale, actions)
    mesh = mesh_factory(shape=(min(2, jax.local_device_count()),))
    with jax.set_mesh(mesh):
        spec = jax.P(mesh.axis_names[0])
        inputs = jax.tree.map(
            lambda x: jax.device_put(x, spec), (loc, raw_scale, actions)
        )
        mapped = jax.jit(
            jax.shard_map(
                per_member, in_specs=(spec, spec, spec), out_specs=(spec, (spec, spec))
            )
        )
        executable = mapped.lower(*inputs).compile()
        actual = executable(*inputs)
    chex.assert_trees_all_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert not compiled_collectives(executable.as_text())


def test_epsilon_greedy_probabilities_and_typed_key_sampling():
    import distrax

    qs = jnp.array([[3.0, 3.0, 0.0], [0.0, 1.0, 0.0]])
    dist = distrax.EpsilonGreedy(qs, 0.3)
    chex.assert_trees_all_close(
        dist.probs,
        jnp.array([[0.45, 0.45, 0.1], [0.1, 0.8, 0.1]]),
    )
    actions = jax.jit(lambda key: dist.sample(seed=key))(jax.random.key(0))
    assert actions.shape == (2,)
    assert jnp.all((actions >= 0) & (actions < 3))
    assert jnp.array_equal(dist.mode(), jnp.array([0, 1]))
    uniform = distrax.EpsilonGreedy(qs, 1.0).probs
    chex.assert_trees_all_close(uniform, jnp.full_like(qs, 1 / 3))
