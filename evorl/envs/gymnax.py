
import jax
import jax.numpy as jnp
from .env import EnvAdapter

import chex
from .space import Space, Box, Discrete
from evorl.types import EnvState, Action

from gymnax.environments.environment import Environment as GymnaxEnv
from .wrappers.gymnax_mod import GymnaxToBraxWrapper
from gymnax.environments.spaces import (
    Box as gymnaxBox,
    Discrete as gymnaxDiscrete,
)


# WIP: add more Space handling support
class GymnaxWrapper(EnvAdapter):
    def __init__(self, env: GymnaxEnv):
        self.env = GymnaxToBraxWrapper(env)
        # TODO: add vmapwrapper etc.

    def reset(self, rng: chex.PRNGKey) -> EnvState:
        return self.env.reset(rng)

    def step(self, state: EnvState, action: Action) -> EnvState:
        return self.env.step(state, action)

    @property
    def action_space(self, state: EnvState) -> Space:
        env_params = state.info["_env_params"]
        return gymnax_space_to_evorl_space(
            self.env.env.action_space(env_params)
        )

    @property
    def obs_space(self, state: EnvState) -> Space:
        env_params = state.info["_env_params"]
        return gymnax_space_to_evorl_space(
            self.env.env.observation_space(env_params)
        )


def gymnax_space_to_evorl_space(space):
    if isinstance(space, gymnaxBox):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gymnaxDiscrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")
