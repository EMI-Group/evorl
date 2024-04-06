import chex
from flax import struct

from .space import Space
from evorl.types import PyTreeDict, EnvLike, EnvInternalState, Action, Observation, Reward, Done, pytree_field
from typing import Tuple

from abc import ABC, abstractmethod


@struct.dataclass
class EnvState:
    """
    Include all the information needed to represent the state of the environment.
    
    """
    env_state: EnvInternalState
    obs: Observation
    reward: Reward
    done: Done
    info: PyTreeDict = pytree_field(default_factory=PyTreeDict)


class Env(ABC):
    """Unified EvoRL Env API"""

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> EnvState:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: EnvState, action: Action) -> EnvState:
        """
            EnvState should have fields like obs, reward, done, info, ...
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """Return the action space of the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_space(self) -> Space:
        """Return the observation space of the environment."""
        raise NotImplementedError


class EnvAdapter(Env):
    """
        Convert envs from other packages to EvoRL's Env API.
    """

    def __init__(self, env: EnvLike):
        self.env = env
