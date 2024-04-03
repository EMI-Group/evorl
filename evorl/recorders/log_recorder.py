import logging
from .recorder import Recorder
from typing import Mapping, Any, Optional

from pprint import pformat

class LogRecorder(Recorder):
    """Log file recorder"""

    def __init__(self, log_path: str, console: bool = True):
        self.logger = logging.getLogger('LogRecorder')
        if not console:
            self.logger.propagate = False

        self.file_handler = logging.FileHandler(log_path)
        # use hydra logger formatter
        self.file_handler.setFormatter(
            logging.getLogger().handlers[0].formatter
        )
        self.logger.addHandler(self.file_handler)

    def write(self, data: Mapping[str, Any], step: Optional[int] = None) -> None:
        self.logger.info(f'iteration {step}:\n'+pformat(data))

    def close(self) -> None:
        self.file_handler.close()
