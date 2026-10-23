"""SwanLab recorder."""

import json
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .recorder import Recorder
from .recorder_utils import (
    flatten_data,
    histogram_values,
    jsonable,
    normalize_step,
    scalar,
)


class SwanlabRecorder(Recorder):
    """Record metrics and media with SwanLab.

    `path` is passed as `swanlab.init(log_dir=...)`, defaulting to the
    training `output_dir`. The SDK creates `run-<timestamp>-<id>/` there
    for the `.swanlab` data file, configuration, debug logs, and media.
    Online mode also uploads to the server; offline mode retains local
    data. Local persistence can be disabled with `core.skip_store` in
    online mode. User configuration and credentials live separately,
    normally under `~/.swanlab/`, configurable through `SWANLAB_ROOT`.
    `extra_kwargs` can override SDK options such as `log_dir` and `mode`.
    """

    def __init__(
        self, *, project, name, config, tags, path, group=None, **extra_kwargs
    ):
        self.kwargs = {
            "project": project,
            "name": name,
            "config": config,
            "tags": tags,
            "log_dir": str(path),
            "group": group,
            **extra_kwargs,
        }

    def init(self) -> None:
        import swanlab

        self.swanlab = swanlab
        self.run = swanlab.init(**self.kwargs)

    def write(self, data: Mapping[str, Any], step: int) -> None:
        step = normalize_step(step)
        payload = {}
        for name, value in flatten_data(data):
            value = scalar(value)
            if value is None:
                continue
            if isinstance(value, pd.Series):
                samples = histogram_values(value)
                if samples.size:
                    counts, edges = np.histogram(samples, bins=64)
                    chart = self.swanlab.echarts.Bar()
                    chart.add_xaxis([f"{x:g}" for x in edges[:-1]])
                    chart.add_yaxis(name, counts.tolist())
                    payload[name] = chart
            elif isinstance(value, pd.DataFrame):
                chart = self.swanlab.echarts.Table()
                chart.add(
                    [str(c) for c in value.columns],
                    value.astype(object).where(pd.notna(value), None).values.tolist(),
                )
                payload[name] = chart
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[name] = value
            else:
                payload[name] = self.swanlab.Text(
                    json.dumps(jsonable(value), default=str)
                )
        if payload:
            self.run.log(payload, step=step)

    def close(self) -> None:
        if hasattr(self, "run"):
            _, error, _ = sys.exc_info()
            if error is not None:
                self.run.finish(state="crashed", error=str(error))
            else:
                self.run.finish()
