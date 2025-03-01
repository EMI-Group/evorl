from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import chex

from evorl.types import (
    Action,
    Done,
    EnvInternalState,
    Observation,
    PyTreeData,
    PyTreeDict,
    Reward,
    pytree_field,
)

from .space import Space


class EnvState(PyTreeData):
    """State of the environment.

    Include all the data needed to represent the state of the environment.

    Attributes:
        env_state: The internal state of the environment.
        obs: The observation of the environment.
        reward: The reward of the environment.
        done: Whether the environment is done.
        info: Extra info from the environment.
        _internal: Extra internal data for the environment.
    """

    env_state: EnvInternalState
    obs: Observation
    reward: Reward
    done: Done
    info: PyTreeDict = pytree_field(default_factory=PyTreeDict)  # info from env
    _internal: PyTreeDict = pytree_field(
        default_factory=PyTreeDict
    )  # extra data for interal use


class Env(ABC):
    """Unified EvoRL Env API."""

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> EnvState:
        """Reset the environment to initial state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, state: EnvState, action: Action) -> EnvState:
        """Take a step in the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """Get the action space of the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_space(self) -> Space:
        """Get the observation space of the environment."""
        raise NotImplementedError


class EnvAdapter(Env):
    """Base class for an environment adapter.

    Convert envs from other packages to EvoRL's Env API.
    """

    def __init__(self, env: Any):
        self.env = env

    @property
    def unwrapped(self) -> Any:
        return self.env


EnvStepFn = Callable[[EnvState, Action], EnvState]
EnvResetFn = Callable[[chex.PRNGKey], EnvState]
