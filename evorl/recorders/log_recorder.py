import logging
from collections.abc import Mapping
from typing import Any

# from pprint import pformat
import yaml

from .recorder import Recorder

# class SubLoggerFilter(logging.Filter):
#     def filter(self, record):
#         # Only allow log records that have the sub-logger's name
#         return record.name == self.name


class LogRecorder(Recorder):
    """Log file recorder"""

    def __init__(self, log_path: str, console: bool = True):
        self.log_path = log_path
        self.console = console

    def init(self) -> None:
        self.logger = logging.getLogger("LogRecorder")

        self.file_handler = logging.FileHandler(self.log_path)
        # use hydra logger formatter
        self.file_handler.setFormatter(logging.getLogger().handlers[0].formatter)
        # self.file_handler.addFilter(
        #     SubLoggerFilter('LogRecorder'))
        self.logger.addHandler(self.file_handler)

        if not self.console:
            self.logger.propagate = False

    def write(self, data: Mapping[str, Any], step: int | None = None) -> None:
        formatted_data = f"iteration {step}:\n" + yaml.dump(data, indent=2)
        self.logger.info(formatted_data)

    def close(self) -> None:
        self.file_handler.close()
