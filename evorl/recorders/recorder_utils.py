"""Data helpers shared by recorder backends and workflows."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def add_prefix(data: dict, prefix: str):
    """Add prefix to the keys of a dictionary."""
    return {f"{prefix}/{k}": v for k, v in data.items()}


def get_1d_array_statistics(data, histogram=False):
    """Return min, max, mean and optionally a histogram source Series."""
    if data is None:
        res = {"min": None, "max": None, "mean": None}
        if histogram:
            res["val"] = pd.Series(dtype=float)
        return res

    res = {
        "min": np.nanmin(data).tolist(),
        "max": np.nanmax(data).tolist(),
        "mean": np.nanmean(data).tolist(),
    }
    if histogram:
        res["val"] = pd.Series(np.asarray(data).ravel())
    return res


def get_1d_array(data):
    """Return statistics alongside raw values for population evaluation."""
    if data is None:
        return {"min": None, "max": None, "mean": None, "val": []}
    res = get_1d_array_statistics(data)
    res["val"] = data
    return res


def flatten_data(data: Mapping[str, Any], prefix: str = ""):
    """Flatten nested metrics while keeping Series and DataFrame values intact."""
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from flatten_data(value, name)
        else:
            yield name, value


def scalar(value: Any):
    """Convert NumPy and JAX scalar values to Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape") and value.shape == () and hasattr(value, "item"):
        return value.item()
    return value


def normalize_step(step: int) -> int:
    """Require an integer step, accepting NumPy and JAX integer scalars."""
    step = scalar(step)
    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError("step must be an integer")
    return step


def jsonable(value: Any):
    """Convert array values into JSON compatible data without losing elements."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def histogram_values(series: pd.Series) -> np.ndarray:
    """Return finite numeric histogram samples, including an empty Series."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    return values[np.isfinite(values)]


def dataframe_to_json(frame: pd.DataFrame) -> str:
    """Serialize a table snapshot; the backend records step separately."""
    return frame.to_json(orient="table", double_precision=15)
