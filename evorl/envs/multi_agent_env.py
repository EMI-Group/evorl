from abc import abstractmethod
from collections.abc import Mapping

import chex

from evorl.types import Action, AgentID, EnvLike

from .env import Env, EnvState
from .space import Space


class MultiAgentEnv(Env):
    """Unified EvoRL Env API"""

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> EnvState:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: EnvState, action: Mapping[AgentID, Action]) -> EnvState:
        """
        EnvState should have fields like obs, reward, done, info, ...
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> Mapping[AgentID, Space]:
        """Return the action space of the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_space(self) -> Mapping[AgentID, Space]:
        """Return the observation space of the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def agents(self) -> list[AgentID]:
        raise NotImplementedError

    @property
    def unwrapped(self) -> EnvLike:
        return self.env


class MultiAgentEnvAdapter(MultiAgentEnv):
    def __init__(self, env: EnvLike):
        self.env = env
