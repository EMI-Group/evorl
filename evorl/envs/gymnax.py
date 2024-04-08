
import jax
import jax.numpy as jnp

from flax import struct
import chex
from .env import EnvAdapter, EnvState, Env
from .space import Space, Box, Discrete
from .wrappers.action_wrapper import ActionSquashWrapper

from evorl.types import Action, PyTreeDict, pytree_field

import gymnax
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.spaces import (
    Space as GymnaxSpace,
    Box as GymnaxBox,
    Discrete as GymnaxDiscrete,
)
from typing import Any, Dict, Optional

from .wrappers.training_wrapper import EpisodeWrapper, OneEpisodeWrapper, VmapAutoResetWrapper, VmapWrapper, VmapAutoResetWrapperV2


class GymnaxAdapter(EnvAdapter):
    def __init__(self, env: GymnaxEnv, env_params: Optional[chex.ArrayTree] = None):
        super(GymnaxAdapter, self).__init__(env)
        self.env_params = env_params or env.default_params

        self._action_space = gymnax_space_to_evorl_space(
            self.env.action_space(self.env_params)
        )
        self._obs_space = gymnax_space_to_evorl_space(
            self.env.observation_space(self.env_params)
        )

    def reset(self, key: chex.PRNGKey) -> EnvState:
        key, reset_key = jax.random.split(key)
        obs, env_state = self.env.reset(reset_key, self.env_params)

        info = PyTreeDict(
            discount=jnp.ones(()),
            env_params=self.env_params,
            step_key=key,
        )

        return EnvState(
            env_state=env_state,
            obs=obs,
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            info=info
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        key, step_key = jax.random.split(state.info.step_key)

        # call step_env() instead of step() to disable autoreset
        # we handle the autoreset at AutoResetWrapper
        obs, env_state, reward, done, info = self.env.step_env(
            step_key, state.env_state, action, state.info.env_params)
        reward = reward.astype(jnp.float32)
        done = done.astype(jnp.float32)

        state.info.update(info)
        state.info.step_key = key

        return state.replace(
            env_state=env_state,
            obs=obs,
            reward=reward,
            done=done
        )

    @property
    def action_space(self) -> Space:
        return self._action_space

    @property
    def obs_space(self) -> Space:
        return self._obs_space


def gymnax_space_to_evorl_space(space: GymnaxSpace):
    if isinstance(space, GymnaxBox):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, GymnaxDiscrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")


def create_gymnax_env(env_name: str, **kwargs) -> GymnaxAdapter:
    env, env_params = gymnax.make(env_name)

    update_env_params = {
        k: v for k, v in kwargs.items()
        if hasattr(env_params, k)
    }
    env_params = env_params.replace(**update_env_params)

    env = GymnaxAdapter(env, env_params)

    if isinstance(env.action_space, Box):
        if not jnp.logical_and(
            (env.action_space.low == -1).all(),
            (env.action_space.high == 1).all()
        ):
            env = ActionSquashWrapper(env)

    return env


def create_wrapped_gymnax_env(env_name: str,
                              episode_length: int = 1000,
                              parallel: int = 1,
                              autoreset: bool = True,
                              fast_reset: bool = False,
                              discount: float = 1.0,
                              **kwargs) -> Env:
    env = create_gymnax_env(env_name, **kwargs)

    if autoreset:
        env = EpisodeWrapper(env, episode_length,
                             record_episode_return=True, discount=discount)
        if fast_reset:
            env = VmapAutoResetWrapperV2(env, num_envs=parallel)
        else:
            env = VmapAutoResetWrapper(env, num_envs=parallel)
    else:
        env = OneEpisodeWrapper(env, episode_length)
        env = VmapWrapper(env, num_envs=parallel, vmap_step=False)

    return env
