"""Comet recorder."""

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import pandas as pd

from .recorder import Recorder
from .recorder_utils import (
    flatten_data,
    histogram_values,
    jsonable,
    normalize_step,
    scalar,
)


class CometRecorder(Recorder):
    """Record EvoRL runs as Comet experiments."""

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
        import comet_ml

        self.experiment = comet_ml.start(
            **{"project_name": self.project, "mode": "create", **self.extra_kwargs}
        )
        self.experiment.set_name(self.name)
        self.experiment.add_tags(self.tags)
        self.experiment.log_parameters(self.config, nested_support=True)
        if self.group is not None:
            self.experiment.log_other("group", self.group)
        self.experiment.log_other("output_dir", str(self.path))

    def write(self, data: Mapping[str, Any], step: int) -> None:
        step = normalize_step(step)
        # All Comet logs inherit this step, including log_table's assets.
        self.experiment.set_step(step)
        for name, value in flatten_data(data):
            value = scalar(value)
            if value is None:
                continue
            if isinstance(value, pd.Series):
                samples = histogram_values(value)
                if samples.size:
                    self.experiment.log_histogram_3d(values=samples, name=name)
            elif isinstance(value, pd.DataFrame):
                self.experiment.log_table(
                    filename=f"{quote(name, safe='')}.json",
                    tabular_data=value,
                    double_precision=15,
                )
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                self.experiment.log_metric(name, value)
            else:
                self.experiment.log_text(
                    json.dumps(jsonable(value), default=str),
                    metadata={"name": name},
                )

    def close(self) -> None:
        if hasattr(self, "experiment"):
            self.experiment.end()
