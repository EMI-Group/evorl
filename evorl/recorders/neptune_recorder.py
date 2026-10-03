"""Adapter for the discontinued Neptune tracking service."""

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .recorder import Recorder
from .recorder_utils import (
    dataframe_to_json,
    flatten_data,
    histogram_values,
    jsonable,
    normalize_step,
    scalar,
)


class NeptuneRecorder(Recorder):
    """Record through the legacy neptune-scale SDK when a server is available."""

    def __init__(
        self, *, project, name, config, tags, path, group=None, **extra_kwargs
    ):
        self.project = project
        self.name = name
        self.config = config
        self.tags = tags
        self.path = path
        self.group = group
        self.extra_kwargs = extra_kwargs

    def init(self) -> None:
        from neptune_scale import Run
        from neptune_scale.types import Histogram

        self.Histogram = Histogram
        self.run = Run(
            **{
                "project": self.project,
                "experiment_name": self.name,
                "log_directory": str(self.path),
                **self.extra_kwargs,
            }
        )
        self.run.log_configs(
            {"config": self.config}, flatten=True, cast_unsupported=True
        )
        self.run.add_tags(tags=self.tags)
        if self.group is not None:
            self.run.add_tags(tags=[self.group], group_tags=True)

    def write(self, data: Mapping[str, Any], step: int) -> None:
        step = normalize_step(step)
        metrics, histograms, strings = {}, {}, {}
        for name, value in flatten_data(data):
            value = scalar(value)
            if value is None:
                continue
            if isinstance(value, pd.Series):
                samples = histogram_values(value)
                if samples.size:
                    counts, edges = np.histogram(samples, bins=64)
                    histograms[name] = self.Histogram(bin_edges=edges, counts=counts)
            elif isinstance(value, pd.DataFrame):
                strings[name] = dataframe_to_json(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[name] = value
            else:
                strings[name] = json.dumps(jsonable(value), default=str)
        if metrics:
            self.run.log_metrics(data=metrics, step=step)
        if histograms:
            self.run.log_histograms(histograms=histograms, step=step)
        if strings:
            self.run.log_string_series(data=strings, step=step)

    def close(self) -> None:
        if hasattr(self, "run"):
            self.run.close()
