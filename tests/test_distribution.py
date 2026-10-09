import jax
import jax.numpy as jnp
import chex
from evorl.distribution import get_tanh_norm_dist


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
