import chex
from ..types import EnvLike
from .space import Space
from evorl.types import EnvState, Action

from abc import ABC,abstractmethod



class Env(ABC):
    """Unified EvoRL Env API"""

    @abstractmethod
    def reset(self, rng: chex.PRNGKey) -> EnvState:
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

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Return the number of environments."""
        raise NotImplementedError

class EnvAdapter(Env):
    """
        Convert envs from other packages to EvoRL's Env API.
    """
    def __init__(self, env: EnvLike):
        self.env = env

    # def __getattr__(self, name):
    #     if name == '__setstate__':
    #         raise AttributeError(name)
        
    #     if hasattr(self, name) and not hasattr(self.env, name):
    #         return getattr(self, name)
    #     else:
    #         return getattr(self.env, name)