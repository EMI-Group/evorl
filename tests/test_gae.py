import jax
import jax.numpy as jnp
import chex

from evorl.utils.toolkits import compute_gae
from evorl.types import SampleBatch


def test_gae():
    keys = jax.random.split(jax.random.PRNGKey(42), 3)

    T = 11
    B = 7

    compute_gae(
        jnp.zeros((T, B), dtype=jnp.float32),
        jax.random.uniform(keys[1], (T, B), dtype=jnp.float32),
        jax.random.uniform(keys[2], (T+1, B), dtype=jnp.float32),
        0.95,
        0.99
    )
