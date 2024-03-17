import jax
import jax.numpy as jnp
import chex
from evorl.agents.a2c import A2CWorkflow

from hydra import compose, initialize

def test_a2c():
    with initialize(config_path="../configs"):
        cfg = compose(config_name="a2c")
    learner = A2CWorkflow(cfg)
    state = learner.init(jax.random.PRNGKey(42))
    state_1 = learner.step(state)