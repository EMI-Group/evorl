from collections.abc import Mapping
from typing import Any
import warnings

import jax.tree_util as jtu
import numpy as np
import pandas as pd
import wandb

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
            **wandb_kwargs,
        }

    def init(self) -> None:
        wandb.init(**self.wandb_kwargs)

    def write(self, data: Mapping[str, Any], step: int | None = None) -> None:
        data = jtu.tree_map(lambda x: _convert_data(x), data)
        wandb.log(data, step=step)

    def close(self):
        wandb.finish()


def _convert_data(val: Any):
    """
    Special handling of pandas objects for wandb logging
    """
    if isinstance(val, pd.Series):
        return wandb.Histogram(val)
    elif isinstance(val, pd.DataFrame):
        return wandb.Table(dataframe=val)
    else:
        return val


def add_prefix(data: dict, prefix: str):
    return {f"{prefix}/{k}": v for k, v in data.items()}


def get_1d_array_statistics(data, histogram=False):
    if data is None:
        res = dict(min=None, max=None, mean=None)
        if histogram:
            res["val"] = pd.Series(data)
        return res

    nan_mask = np.isnan(data)
    if nan_mask.any():
        warnings.warn("data contains nan, removing them...")
        data = data[~nan_mask]

    res = dict(
        min=np.min(data).tolist(),
        max=np.max(data).tolist(),
        mean=np.mean(data).tolist(),
    )

    if histogram:
        res["val"] = pd.Series(data)

    return res


def get_1d_array(data):
    res = dict(
        min=np.min(data).tolist(),
        max=np.max(data).tolist(),
        mean=np.mean(data).tolist(),
    )

    res["val"] = data

    return res
