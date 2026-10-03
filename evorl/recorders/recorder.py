from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from .recorder_utils import normalize_step


class Recorder(ABC):
    """A Recorder Interface."""

    @abstractmethod
    def init(self) -> None:
        """Initialize the recorder."""
        raise NotImplementedError

    @abstractmethod
    def write(self, data: Mapping[str, Any], step: int) -> None:
        """Write data at an explicit iteration; calls may share the same step."""
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
        self.recorders = list(recorders)

    def add_recorder(self, recorder: Recorder) -> None:
        self.recorders.append(recorder)

    def init(self) -> None:
        for recorder in self.recorders:
            recorder.init()

    def write(self, data: Mapping[str, Any], step: int) -> None:
        step = normalize_step(step)
        for recorder in self.recorders:
            recorder.write(data, step)

    def close(self) -> None:
        first_error = None
        for recorder in self.recorders:
            try:
                recorder.close()
            except Exception as error:  # noqa: BLE001 - re-raised after all closes
                # Every backend needs a chance to flush its pending logs.
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
