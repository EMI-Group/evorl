"""Aim recorder."""

import json
from collections.abc import Mapping
from typing import Any

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


class AimRecorder(Recorder):
    """Record runs in a shared local Aim repository."""

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
        import aim

        self.aim = aim
        # Aim's experiment is the closest equivalent to a run group.
        self.run = aim.Run(
            **{
                "repo": str(self.path),
                "experiment": self.group if self.group is not None else self.project,
                **self.extra_kwargs,
            }
        )
        self.run.name = self.name
        self.run["project"] = self.project
        self.run["config"] = self.config
        for tag in self.tags:
            self.run.add_tag(tag)

    def write(self, data: Mapping[str, Any], step: int) -> None:
        step = normalize_step(step)
        for name, value in flatten_data(data):
            value = scalar(value)
            if value is None:
                continue
            if isinstance(value, pd.Series):
                samples = histogram_values(value)
                if samples.size:
                    self.run.track(
                        self.aim.Distribution(samples=samples),
                        name=name,
                        step=step,
                    )
            elif isinstance(value, pd.DataFrame):
                self.run.track(
                    self.aim.Text(dataframe_to_json(value)), name=name, step=step
                )
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                self.run.track(value, name=name, step=step)
            else:
                self.run.track(
                    self.aim.Text(json.dumps(jsonable(value), default=str)),
                    name=name,
                    step=step,
                )

    def close(self) -> None:
        if hasattr(self, "run"):
            self.run.close()
