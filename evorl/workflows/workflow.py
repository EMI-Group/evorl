import jax
import chex
from omegaconf import DictConfig


from typing import Any, Tuple, Union
from typing_extensions import (
    Self  # pytype: disable=not-supported-yet
)
from abc import ABC, abstractmethod

from evorl.types import State
from evorl.recorders import Recorder, ChainRecorder

# TODO: remove it when evox is updated


class AbstractWorkflow(ABC):
    """
        A duck-type of evox.Workflow without auto recursive setup mechanism.
    """

    @abstractmethod
    def setup(self, key: jax.Array) -> State:
        """
            Custom setup.
            When call public API init(), setup() would be recursively called.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, state: State) -> Tuple[Any, State]:
        raise NotImplementedError

    def learn(self, state: State) -> State:
        """
            run the complete learning process.
            Note: this is designed for the non pure function. Don't wrap it with jit.
        """
        raise NotImplementedError

    def init(self, key: jax.Array) -> State:
        """
            Initialize the state of the module.
            This is the public API to call for instance state initialization.
        """
        self.recorder.init()
        return self.setup(key)

    @classmethod
    def name(cls) -> str:
        """
            Name of the workflow(eg. PPO, PSO, etc.)
            Default is the Workflow class name.
        """
        return cls.__name__


class Workflow(AbstractWorkflow):
    def __init__(
        self,
        config: DictConfig
    ):
        self.config = config
        self.recorder = ChainRecorder()

    @classmethod
    def build_from_config(cls, config: DictConfig, *args, **kwargs) -> Self:
        """
            Build the workflow from the config.
        """
        raise NotImplementedError

    def add_recorders(self, recorders: Recorder) -> None:
        for recorder in recorders:
            self.recorder.add_recorder(recorder)

    def close(self) -> None:
        """
            Close the workflow's components.
        """
        self.recorder.close()

    def learn(self, state: State) -> State:
        """
            Run the complete learning process:
                - call multiple times of step()
                - record the metrics
                - save checkpoints
        """
        raise NotImplementedError