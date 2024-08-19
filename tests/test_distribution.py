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
        jax.random.PRNGKey(42), shape=(T, B, A), minval=-0.999, maxval=0.999
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
        jax.random.PRNGKey(42), shape=(T, B, A), minval=-1.0, maxval=1.0
    )

    def loss_fn(loc, scale):
        actions_dist = get_tanh_norm_dist(loc, scale)
        logp = actions_dist.log_prob(actions)

        return -logp.mean()

    loss, (g_loc, g_scale) = jax.value_and_grad(loss_fn, argnums=(0, 1))(loc, scale)

    assert not jnp.isnan(g_loc).any(), "loc grad has nan"
    assert not jnp.isnan(g_scale).any(), "scale grad has nan"
