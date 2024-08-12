from collections.abc import Mapping
from typing import Any
import wandb
import numpy as np
import pandas as pd

import jax.tree_util as jtu

from .recorder import Recorder


class WandbRecorder(Recorder):
    def __init__(
        self, *, project, name, config, tags, path, mode="disabled", **wandb_kwargs
    ):
        self.wandb_kwargs = {
            "project": project,
            "name": name,
            "config": config,
            "tags": tags,
            "dir": path,
            "mode": mode,
            "settings": wandb.Settings(start_method="thread"),
            **wandb_kwargs,
        }

    def init(self) -> None:
        wandb.init(**self.wandb_kwargs)

    def write(self, data: Mapping[str, Any], step: int | None = None) -> None:
        data = jtu.tree_map(lambda x: _convert_2d_data(x), data)
        wandb.log(data, step=step)

    def close(self):
        wandb.finish()


def _convert_2d_data(val):
    if isinstance(val, np.ndarray) and val.ndim == 1:
        return wandb.Histogram(val)
    elif isinstance(val, pd.DataFrame):
        return wandb.Table(dataframe=val)
    else:
        return val
