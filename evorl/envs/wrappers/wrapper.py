from flax import struct
import chex
from evorl.types import ExtraInfo, PyTreeDict, Action
from ..env import Env, EnvState

from typing import Tuple, Union



class Wrapper(Env):
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env: Env):
        self.env = env

    def reset(self, key: chex.PRNGKey) -> EnvState:
        return self.env.reset(key)

    def step(self, state: EnvState, action: Action) -> EnvState:
        return self.env.step(state.env_state, action)

    @property
    def obs_space(self) -> int:
        return self.env.obs_space

    @property
    def action_space(self) -> int:
        return self.env.action_space

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)
