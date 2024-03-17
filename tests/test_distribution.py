import jax
import jax.numpy as jnp
import chex
from evorl.utils.distribution import TanhNormal

from .utils import disable_gpu_preallocation

def test_tanh_normal():
    disable_gpu_preallocation()

    T=11
    B=7
    A=3

    loc = jnp.zeros((T, B, A))
    scale = jnp.ones((T, B, A))

    actions_dist = TanhNormal(loc, scale)

    
    actions = jax.random.uniform(jax.random.PRNGKey(42), shape=(T, B, A), minval=-0.999, maxval=0.999)

    logp = actions_dist.log_prob(actions)

    chex.assert_shape(logp, (T, B))

    



def test_tanh_normal_grad():
    disable_gpu_preallocation()

    T=32
    B=8
    A=7

    loc = jnp.zeros((T, B, A))
    scale = jnp.ones((T, B, A))

    actions = jax.random.uniform(jax.random.PRNGKey(42), shape=(T, B, A), minval=-1.0, maxval=1.0)

    def loss_fn(loc, scale):
        actions_dist = TanhNormal(loc, scale)
        logp = actions_dist.log_prob(actions)

        return -logp.mean()

    loss, (g_loc, g_scale) = jax.value_and_grad(loss_fn, argnums=(0,1))(loc, scale)

    assert not jnp.isnan(g_loc).any(), "loc grad has nan"
    assert not jnp.isnan(g_scale).any(), "scale grad has nan"
