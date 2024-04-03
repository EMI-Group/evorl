import chex
from .env import Env, EnvState
from evorl.types import (
    EnvLike,
    Space,
    Action,
)
from typing import Mapping
from abc import abstractmethod

class MultiAgentEnv(Env):
    """Unified EvoRL Env API"""

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> EnvState:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: EnvState, action: Mapping[str,Action]) -> EnvState:
        """
            EnvState should have fields like obs, reward, done, info, ...
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> Mapping[str,Space]:
        """Return the action space of the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_space(self) -> Mapping[str,Space]:
        """Return the observation space of the environment."""
        raise NotImplementedError

class MultiAgentEnvAdapter(MultiAgentEnv):
    def __init__(self, env: EnvLike):
        self.env = env