from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


class Recorder(ABC):
    """A Recorder Interface."""

    @abstractmethod
    def init(self) -> None:
        """Initialize the recorder."""
        raise NotImplementedError

    @abstractmethod
    def write(self, data: Mapping[str, Any], step: int | None = None) -> None:
        """Write data to the recorder."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Finalize the recorder."""
        raise NotImplementedError


class ChainRecorder(Recorder):
    """Container for multiple recorders."""

    def __init__(self, recorders: Sequence[Recorder]):
        """Initialize the ChainRecorder.

        Args:
            recorders: A sequence of recorders to use.
        """
        self.recorders = recorders

    def add_recorder(self, recorder: Recorder) -> None:
        self.recorders.append(recorder)

    def init(self) -> None:
        for recorder in self.recorders:
            recorder.init()

    def write(self, data: Mapping[str, Any], step: int | None = None) -> None:
        for recorder in self.recorders:
            recorder.write(data, step)

    def close(self) -> None:
        for recorder in self.recorders:
            recorder.close()
