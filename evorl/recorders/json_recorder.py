import json
import logging
from pathlib import Path
from typing import Any, Optional
from collections.abc import Mapping

from .recorder import Recorder


# TODO: test it
class JsonRecorder(Recorder):
    """Log file recorder"""

    def __init__(self, path: str):
        self.path = path

    def init(self) -> None:
        path = Path(self.path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        self.f = path.open("a")

    def write(self, data: Mapping[str, Any], step: int | None = None) -> None:
        json.dump((dict(step=step), data), self.f, indent=4)

    def close(self) -> None:
        self.f.close()
