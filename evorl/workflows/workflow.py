from abc import ABC, abstractmethod
from typing import Any

import chex
from omegaconf import DictConfig
from typing_extensions import Self  # pytype: disable=not-supported-yet

from evorl.recorders import ChainRecorder, Recorder
from evorl.types import State
from evorl.utils.orbax_utils import setup_checkpoint_manager

# TODO: remove it when evox is updated


class AbstractWorkflow(ABC):
    """A Workflow Interface for EvoRL training pipelines."""

    @abstractmethod
    def init(self, key: chex.PRNGKey) -> State:
        """Initialize the workflow's state.

        Args:
            key: JAX PRNGKey

        Returns:
            state: the state of the workflow
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, state: State) -> tuple[Any, State]:
        """Define the logic of one training iteration."""
        raise NotImplementedError

    @classmethod
    def name(cls) -> str:
        """Define the name of the workflow (eg. PPO, PSO, etc.).

        Default workflow name is its class name.
        """
        return cls.__name__


class Workflow(AbstractWorkflow):
    """The base class for all Workflows.

    All workflow classes are inherit from this class, and customize by implementing
    """

    def __init__(self, config: DictConfig):
        """Initialize a RLWorkflow instance.

        Args:
            config: the config object.
        """
        self.config = config
        self.recorder = ChainRecorder([])
        self.checkpoint_manager = setup_checkpoint_manager(config)

    @classmethod
    def build_from_config(cls, config: DictConfig, *args, **kwargs) -> Self:
        """Build the workflow instance from the config.

        This is the public API to call for instantiating a new workflow object from config. Normally, it will call __init__() and do some pre- and post-processing.

        Args:
            config: config object

        Returns:
            A workflow instance

        """
        raise NotImplementedError

    def init(self, key: chex.PRNGKey) -> State:
        """Initialize the state of the .

        This is the public API to call for instance state initialization.
        """
        self.recorder.init()
        state = self.setup(key)
        return state

    def setup(self, key: chex.PRNGKey) -> State:
        raise NotImplementedError

    def add_recorders(self, recorders: Recorder) -> None:
        for recorder in recorders:
            self.recorder.add_recorder(recorder)

    def close(self) -> None:
        """Close the workflow's components."""
        self.recorder.close()
        self.checkpoint_manager.close()

    def learn(self, state: State) -> State:
        """Run the complete learning process.

        The learning process includes:

        - call multiple times of step()
        - record the metrics
        - save checkpoints
        """
        raise NotImplementedError
