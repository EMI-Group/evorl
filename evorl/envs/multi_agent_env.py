from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import chex

from evorl.types import Action, AgentID

from .env import Env, EnvState
from .space import Space


class MultiAgentEnv(Env):
    """Unified EvoRL Multi-Agent Env API."""

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> EnvState:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: EnvState, action: Mapping[AgentID, Action]) -> EnvState:
        """EnvState should have fields like obs, reward, done, info, ..."""
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


class MultiAgentEnvAdapter(MultiAgentEnv):
    """Base class for an multi-agent environment adapter.

    Convert envs from other packages to EvoRL's MultiAgentEnv API.
    """

    def __init__(self, env: Any):
        self.env = env

    @property
    def unwrapped(self) -> Any:
        return self.env
