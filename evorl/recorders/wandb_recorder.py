import sys
from collections.abc import Mapping
from typing import Any

import jax.tree_util as jtu
import pandas as pd

from .recorder import Recorder

# Backwards-compatible helper exports for direct module imports.
from .recorder_utils import (  # noqa: F401
    add_prefix,
    get_1d_array,
    get_1d_array_statistics,
    normalize_step,
)


class WandbRecorder(Recorder):
    """Recorder for Weights & Biases."""

    def __init__(
        self, *, project, name, config, tags, path, group=None, **extra_kwargs
    ):
        self.wandb_kwargs = {
            "project": project,
            "name": name,
            "config": config,
            "tags": tags,
            "dir": path,
            "group": group,
            **extra_kwargs,
        }

    def init(self) -> None:
        import wandb

        self.wandb = wandb
        self.run = wandb.init(**self.wandb_kwargs)

    def write(self, data: Mapping[str, Any], step: int) -> None:
        step = normalize_step(step)
        data = jtu.tree_map(self._convert_data, data)
        self.run.log(data, step=step)

    def close(self):
        if not hasattr(self, "run"):
            return
        ext_type, _, _ = sys.exc_info()
        if ext_type is not None:
            self.run.finish(exit_code=1)
        else:
            self.run.finish()

    def _convert_data(self, val: Any):
        if isinstance(val, pd.Series):
            return self.wandb.Histogram(val)
        if isinstance(val, pd.DataFrame):
            return self.wandb.Table(dataframe=val)
        return val
