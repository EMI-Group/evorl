import wandb
from .recorder import Recorder
from typing import Mapping, Any, Optional

class WandbRecorder(Recorder):
    def __init__(self, *, project, name, config, tags, dir, mode='disabled', **kwargs):
        wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            dir=dir,
            mode=mode,
            **kwargs
        )

    def write(self, data: Mapping[str, Any], step: Optional[int] = None) -> None:
        wandb.log(data, step=step)

    def close(self):
        wandb.finish()
