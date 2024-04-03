
from abc import ABC, abstractmethod
from typing import Mapping, Any, Sequence, Optional


class Recorder(ABC):
    @abstractmethod
    def write(self, data: Mapping[str, Any], step: Optional[int] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class ChainRecorder(Recorder):
    def __init__(self, recorders: Sequence[Recorder]):
        self.recorders = recorders

    def add_recorder(self, recorder: Recorder) -> None:
        self.recorders.append(recorder)

    def write(self, data: Mapping[str, Any], step: Optional[int] = None) -> None:
        for recorder in self.recorders:
            recorder.write(data, step)

    def close(self) -> None:
        for recorder in self.recorders:
            recorder.close()
