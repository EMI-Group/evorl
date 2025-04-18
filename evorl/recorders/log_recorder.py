import logging
from collections.abc import Mapping
from typing import Any

import jax.tree_util as jtu
import numpy as np
import pandas as pd

# from pprint import pformat
import yaml

from .recorder import Recorder

# class SubLoggerFilter(logging.Filter):
#     def filter(self, record):
#         # Only allow log records that have the sub-logger's name
#         return record.name == self.name


class LogRecorder(Recorder):
    """Log file recorder."""

    def __init__(self, log_path: str, console: bool = True):
        """Initialize the log recorder.

        Args:
            log_path: The path to the log file.
            console: Whether to print the log to the console. Defaults to True.
        """
        self.log_path = log_path
        self.console = console

    def init(self) -> None:
        self.logger = logging.getLogger("LogRecorder")

        self.file_handler = logging.FileHandler(self.log_path, mode="w")
        # use root logger formatter (usually set by hydra)
        self.file_handler.setFormatter(logging.getLogger().handlers[0].formatter)
        self.logger.addHandler(self.file_handler)

        if not self.console:
            self.logger.propagate = False

    def write(self, data: Mapping[str, Any], step: int | None = None) -> None:
        data = jtu.tree_map(lambda x: _convert_data(x), data)
        formatted_data = f"iteration {step}:\n" + yaml.dump(data, indent=2)
        self.logger.info(formatted_data)

    def close(self) -> None:
        self.file_handler.close()


def _convert_data(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, np.generic):
        return val.item()
    elif isinstance(val, pd.Series) or isinstance(val, pd.DataFrame):
        # escape the special data for wandb
        return None
    else:
        return val
