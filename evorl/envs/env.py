from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import jax

from evorl.types import (
    Action,
    Done,
    EnvAction,
    EnvInternalState,
    EnvObservation,
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
        done: Whether the environment is done. Multi-agent environments use an
            agent-keyed mapping with `__all__` indicating the episode boundary.
        info: Extra info from the environment.
        _internal: Extra internal data for the environment.
    """

    env_state: EnvInternalState
    obs: EnvObservation
    reward: Reward
    done: Done
    info: PyTreeDict = pytree_field(default_factory=PyTreeDict)  # info from env
    _internal: PyTreeDict = pytree_field(
        default_factory=PyTreeDict
    )  # extra data for interal use


class BaseEnv[ActionT, SpaceT](ABC):
    """Common functional interface for environments and their wrappers.

    Instances hold static configuration; evolving data is passed explicitly in
    `EnvState`. Implementations choose their action and space representations
    through `ActionT` and `SpaceT`. Wrappers can preserve these types without
    depending on a particular environment interface.
    """

    @abstractmethod
    def reset(self, key: jax.Array) -> EnvState:
        """Reset the environment to initial state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, state: EnvState, action: ActionT) -> EnvState:
        """Take a step in the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> SpaceT:
        """Get the action space of the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_space(self) -> SpaceT:
        """Get the observation space of the environment."""
        raise NotImplementedError


class Env(BaseEnv[Action, Space]):
    """Single-agent environment with one observation and action space.

    Structured observations and actions are represented by a `SpaceContainer`,
    which is itself a `Space`, rather than an agent-keyed mapping of spaces.
    """


class EnvAdapter(Env):
    """Base class for an environment adapter.

    Convert envs from other packages to EvoRL's Env API.
    """

    def __init__(self, env: Any):
        self.env = env

    @property
    def unwrapped(self) -> Any:
        return self.env


EnvStepFn = Callable[[EnvState, EnvAction], EnvState]
EnvResetFn = Callable[[jax.Array], EnvState]


def get_episode_done(done: Done) -> jax.Array:
    """Get the episode boundary from single-agent or multi-agent done flags."""
    return done["__all__"] if isinstance(done, Mapping) else done
