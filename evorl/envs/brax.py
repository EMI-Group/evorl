import jax.numpy as jnp
from .env import EnvAdapter

import chex
from .space import Space, Box
from evorl.types import EnvState, Action

import brax.envs
from .wrappers.brax_mod import (
    EpisodeWrapper, 
    EpisodeWrapperV2,
    AutoResetWrapper, 
    VmapWrapper
)

class BraxEnvAdapter(EnvAdapter):
    def reset(self, rng: chex.PRNGKey) -> EnvState:
        return self.env.reset(rng)

    def step(self, state: EnvState, action: Action) -> EnvState:
        return self.env.step(state, action)

    @property
    def action_space(self) -> Space:
        # ref: brax's GymWrapper
        action_spec = self.env.sys.actuator.ctrl_range
        action_spec = action_spec.astype(jnp.float32)
        return Box(low=action_spec[:, 0], high=action_spec[:, 1])

    @property
    def obs_space(self) -> Space:
        # ref: brax's GymWrapper
        obs_spec = jnp.full((self.env.observation_size,),
                            jnp.inf, dtype=jnp.float32)
        return Box(low=-obs_spec, high=obs_spec)


def create_env(env_name: str,
               episode_length: int = 1000,
               action_repeat: int = 1,
               parallel: int = 1,
               autoset: bool = True,
               **kwargs):
    env = brax.envs.get_environment(env_name, **kwargs)

    if autoset:
        env = EpisodeWrapper(env, episode_length, action_repeat)
        env = VmapWrapper(env, batch_size=parallel)
        env = AutoResetWrapper(env)
    else:
        env = EpisodeWrapperV2(env, episode_length, action_repeat)
        env = VmapWrapper(env, batch_size=parallel)
    
    # To EvoRL Env
    env = BraxEnvAdapter(env)

    return env
