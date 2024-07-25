from abc import ABC, abstractmethod
from typing import Any, Optional
from collections.abc import Mapping, Sequence


class Recorder(ABC):
    @abstractmethod
    def init(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def write(self, data: Mapping[str, Any], step: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class ChainRecorder(Recorder):
    def __init__(self, recorders: Sequence[Recorder] = []):
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
