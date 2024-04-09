import logging
from .recorder import Recorder
from typing import Mapping, Any, Optional

from pathlib import Path

import json

#TODO: test it
class JsonRecorder(Recorder):
    """Log file recorder"""

    def __init__(self, path: str):
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        self.f = path.open('a')

    def write(self, data: Mapping[str, Any], step: Optional[int] = None) -> None:
        json.dump((dict(step=step), data), self.f, indent=4)

    def close(self) -> None:
        self.f.close()
