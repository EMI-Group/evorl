import logging
from collections.abc import Mapping
from typing import Any

import jax.tree_util as jtu
import numpy as np
import pandas as pd
import yaml

from .recorder import Recorder
from .recorder_utils import normalize_step


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
        self.logger = logging.getLogger(f"LogRecorder.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = self.console

        self.file_handler = logging.FileHandler(self.log_path, mode="w")
        # use root logger formatter (usually set by hydra)
        root_handlers = logging.getLogger().handlers
        if root_handlers:
            self.file_handler.setFormatter(root_handlers[0].formatter)
        self.logger.addHandler(self.file_handler)

    def write(self, data: Mapping[str, Any], step: int) -> None:
        step = normalize_step(step)
        data = jtu.tree_map(lambda x: _convert_data(x), data)
        formatted_data = f"iteration {step}:\n" + yaml.dump(data, indent=2)
        self.logger.info(formatted_data)

    def close(self) -> None:
        if hasattr(self, "file_handler"):
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()


def _convert_data(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, np.generic):
        return val.item()
    elif isinstance(val, (pd.Series, pd.DataFrame)):
        # Rich data is handled by the tracking backends.
        return None
    else:
        return val
