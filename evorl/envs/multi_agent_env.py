from abc import abstractmethod
from typing import Any

from evorl.types import Action, AgentID, AgentMapping

from .env import BaseEnv
from .space import Space


class MultiAgentEnv(BaseEnv[AgentMapping[Action], AgentMapping[Space]]):
    """Functional environment with simultaneous actions from all agents.

    `agents` is a fixed list for the lifetime of the environment. Actions,
    observations and spaces are mappings keyed by these IDs; each observation
    may itself be a structured pytree. Rewards and done flags are scalar arrays
    per agent before vectorization. `done["__all__"]` is required and denotes
    the episode boundary used by episode limits and autoreset. An individual
    agent finishing does not reset the environment.

    A native shared reward is exposed unchanged to every agent. An optional
    `reward["__all__"]` can preserve an explicitly provided team reward; adapters
    must not invent one by summing individual rewards. Shared observations for
    centralized training can be provided in `info.global_obs`.
    It follows the current observations across autoreset; when terminal
    observations are recorded, `info.ori_global_obs` preserves its final value.

    Native termination and truncation flags, when available, follow the same
    mapping structure as `done` in `EnvState._internal`. The training wrappers
    expose these as `info.termination` and `info.truncation`. Episode step counts
    and `info.autoreset` describe the whole environment, while episode returns
    follow the reward mapping. Finished agents retain their keys and continue
    to receive an action entry, using the task's inactive/no-op convention.
    """

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
