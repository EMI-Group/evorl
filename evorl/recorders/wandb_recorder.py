from collections.abc import Mapping
from typing import Any

import jax.tree_util as jtu
import numpy as np
import pandas as pd
import wandb

from .recorder import Recorder


class WandbRecorder(Recorder):
    """Recorder for Weights & Biases."""

    def __init__(self, *, project, name, config, tags, path, **wandb_kwargs):
        self.wandb_kwargs = {
            "project": project,
            "name": name,
            "config": config,
            "tags": tags,
            "dir": path,
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
    if isinstance(val, pd.Series):
        return wandb.Histogram(val)
    elif isinstance(val, pd.DataFrame):
        return wandb.Table(dataframe=val)
    else:
        return val


def add_prefix(data: dict, prefix: str):
    """Add prefix to the keys of a dictionary."""
    return {f"{prefix}/{k}": v for k, v in data.items()}


def get_1d_array_statistics(data, histogram=False):
    """Get raw value and statistics of a 1D array.

    Helper function for logging in WandB.

    Args:
        data: 1D numpy array. If data has multiple dimensions, it will be viewed as flattened.
        histogram: If True, return raw data in `pd.Series`, which will be futher converted to histogram in `WandBRecorder`.

    Returns:
        A dictionary containing min, max, mean, and optional raw data.
    """
    if data is None:
        res = dict(min=None, max=None, mean=None)
        if histogram:
            res["val"] = pd.Series()
        return res

    res = dict(
        min=np.nanmin(data).tolist(),
        max=np.nanmax(data).tolist(),
        mean=np.nanmean(data).tolist(),
    )

    if histogram:
        res["val"] = pd.Series(data)

    return res


def get_1d_array(data):
    """Get statistics of a 1D array.

    Similar to `get_1d_array_statistics`, but instead of recording histogram, WandB will record the raw data.
    """
    if data is None:
        res = dict(min=None, max=None, mean=None, val=[])
        return res

    res = dict(
        min=np.nanmin(data).tolist(),
        max=np.nanmax(data).tolist(),
        mean=np.nanmean(data).tolist(),
    )

    res["val"] = data

    return res
