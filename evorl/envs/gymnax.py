
import jax
import jax.numpy as jnp
from .env import EnvAdapter
from flax import struct
import chex
from .space import Space, Box, Discrete
from evorl.types import EnvState, Action, EnvLike

import gymnax
from gymnax.environments.environment import Environment as GymnaxEnv
from .wrappers.gymnax_mod import GymnaxToBraxWrapper
from gymnax.environments.spaces import (
    Box as gymnaxBox,
    Discrete as gymnaxDiscrete,
)
from typing import Any, Dict, Optional
from .wrappers.gymnax_mod import AutoResetWrapper
from .wrappers.brax_mod import (
    EpisodeWrapper,
    EpisodeWrapperV2,
    VmapWrapper,
    EpisodeRecordWrapper,
    get_wrapper
)


@struct.dataclass
class State:
    """Environment state for training and inference."""

    env_state: chex.ArrayTree
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: Dict[str, Any] = struct.field(default_factory=dict)


class GymnaxAdapter(EnvAdapter):
    def __init__(self, env: EnvLike):
        self.env = env
        self.gymnax_env: GymnaxEnv = env.unwrapped.env
        self.env_params = env.env_params

    def reset(self, rng: chex.PRNGKey) -> EnvState:
        return self.env.reset(rng)

    def step(self, state: EnvState, action: Action) -> EnvState:
        return self.env.step(state, action)

    @property
    def action_space(self) -> Space:
        env_params = self.env.env_params
        return gymnax_space_to_evorl_space(
            self.gymnax_env.action_space(env_params)
        )

    @property
    def obs_space(self) -> Space:
        env_params = self.env.env_params
        return gymnax_space_to_evorl_space(
            self.gymnax_env.observation_space(env_params)
        )

    @property
    def num_envs(self) -> int:
        vmap_wrapper = get_wrapper(self.env, VmapWrapper)
        if vmap_wrapper is None:
            return 1
        else:
            return vmap_wrapper.num_envs


def gymnax_space_to_evorl_space(space):
    if isinstance(space, gymnaxBox):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gymnaxDiscrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")


def create_gymnax_env(env_name: str,
                      action_repeat: int = 1,
                      parallel: int = 1,
                      autoreset: bool = True,
                      discount: float = 1.0,
                      **kwargs) -> GymnaxAdapter:
    env, env_params = gymnax.make(env_name)

    update_env_params = {
        k: v for k, v in kwargs.items()
        if hasattr(env_params, k)
    }
    env_params = env_params.replace(**update_env_params)

    # To Brax Env
    env = GymnaxToBraxWrapper(env, env_params)

    episode_length = env_params.max_steps_in_episode

    if autoreset:
        env = EpisodeWrapper(env, episode_length, action_repeat)
        env = EpisodeRecordWrapper(env, discount=discount)
        env = VmapWrapper(env, num_envs=parallel)
        env = AutoResetWrapper(env)
    else:
        env = EpisodeWrapperV2(env, episode_length, action_repeat)
        env = VmapWrapper(env, num_envs=parallel)

    env = GymnaxAdapter(env)
    return env
