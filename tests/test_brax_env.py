import jax
import jax.numpy as jnp
import chex
from evorl.envs import create_brax_env
from evorl.envs.wrappers.brax_mod import (
    has_wrapper, 
    AutoResetWrapper,
    EpisodeWrapper,
    EpisodeWrapperV2
)

def test_brax_env():
    env = create_brax_env("ant")
    action_space = env.action_space
    obs_space = env.obs_space


def test_has_brax_wrapper():
    env = create_brax_env("ant", autoset=True)

    assert has_wrapper(env.env, EpisodeWrapper)
    assert has_wrapper(env.env, AutoResetWrapper)

    env = create_brax_env("ant", autoset=False)
    assert has_wrapper(env.env, EpisodeWrapperV2)
    assert not has_wrapper(env.env, AutoResetWrapper)
    