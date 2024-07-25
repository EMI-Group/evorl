from collections.abc import Mapping
from typing import Any

import wandb

from .recorder import Recorder


class WandbRecorder(Recorder):
    def __init__(
        self, *, project, name, config, tags, path, mode="disabled", **wandb_kwargs
    ):
        self.project = project
        self.name = name
        self.config = config
        self.tags = tags
        self.dir = path
        self.mode = mode
        self.wandb_kwargs = wandb_kwargs

    def init(self) -> None:
        wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            tags=self.tags,
            dir=self.dir,
            mode=self.mode,
            **self.wandb_kwargs
        )

    def write(self, data: Mapping[str, Any], step: int | None = None) -> None:
        wandb.log(data, step=step)

    def close(self):
        wandb.finish()
