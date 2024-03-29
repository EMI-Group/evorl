import jax.numpy as jnp
from .env import EnvAdapter

import chex
from .space import Space, Box
from evorl.types import EnvState, Action

import brax.envs
from .wrappers.brax_mod import (
    EpisodeWrapper, 
    EpisodeWrapperV2,
    EpisodeRecordWrapper,
    AutoResetWrapper, 
    VmapWrapper,
    get_wrapper
)

class BraxEnvAdapter(EnvAdapter):
    def __init__(self, env):
        super(BraxEnvAdapter, self).__init__(env)

        action_spec = self.env.sys.actuator.ctrl_range
        action_spec = action_spec.astype(jnp.float32)
        self._action_sapce = Box(low=action_spec[:, 0], high=action_spec[:, 1])

        obs_spec = jnp.full((self.env.observation_size,),
                            jnp.inf, dtype=jnp.float32)
        self._obs_space = Box(low=-obs_spec, high=obs_spec)

    def reset(self, rng: chex.PRNGKey) -> EnvState:
        return self.env.reset(rng)

    def step(self, state: EnvState, action: Action) -> EnvState:
        return self.env.step(state, action)

    @property
    def action_space(self) -> Space:
        return self._action_sapce

    @property
    def obs_space(self) -> Space:
        return self._obs_space
    
    @property
    def num_envs(self) -> int:
        vmap_wrapper = get_wrapper(self.env, VmapWrapper)
        if vmap_wrapper is None:
            return 1
        else:
            return vmap_wrapper.num_envs


def create_brax_env(env_name: str,
               episode_length: int = 1000,
               action_repeat: int = 1,
               parallel: int = 1,
               autoreset: bool = True,
               discount: float = 1.0,
               **kwargs)-> BraxEnvAdapter:
    """
        Args:
            Autoreset: When use envs for RL training, set autoreset=True. When use envs for evaluation, set autoreset=False.
            discount: discount factor for episode return calculation. The episode returns are Only recorded when autoreset=True.
    """

    env = brax.envs.get_environment(env_name, **kwargs)

    if autoreset:
        env = EpisodeWrapper(env, episode_length, action_repeat)
        env = EpisodeRecordWrapper(env, discount=discount)
        env = VmapWrapper(env, num_envs=parallel)
        env = AutoResetWrapper(env)
    else:
        env = EpisodeWrapperV2(env, episode_length, action_repeat)
        env = VmapWrapper(env, num_envs=parallel)
    
    # To EvoRL Env
    env = BraxEnvAdapter(env)

    return env
