"""Build EvoRL's recorders from the training configuration."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


def setup_recorders(
    config: DictConfig,
    workflow_name: str,
    *,
    parallel: bool = False,
    recorder_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
):
    """Build recorders for train.py or train_dist.py (parallel=True).

    Grouping is a script convention: ordinary runs use "dev", while parallel
    training runs use their experiment name. Backend options can override it.
    This flag describes the entrypoint, not the number of JAX devices or jobs.
    """
    # train_dist must select its GPU before importing EvoRL/JAX.
    from evorl.recorders import (
        AimRecorder,
        CometRecorder,
        LogRecorder,
        NeptuneRecorder,
        SwanlabRecorder,
        WandbRecorder,
    )

    output_dir = Path(config.output_dir)
    tags = OmegaConf.to_container(config.tags, resolve=True)
    exp_name = f"{workflow_name}_{config.env.env_name}_{config.env.env_type}"
    if tags:
        exp_name += "|" + ",".join(tags)
    group = exp_name if parallel else "dev"

    backend_tags = [workflow_name, config.env.env_name, config.env.env_type] + tags
    config_data = OmegaConf.to_container(config, resolve=True)
    backend_options = recorder_kwargs or {}
    recorder_types = {
        "wandb": WandbRecorder,
        "aim": AimRecorder,
        "swanlab": SwanlabRecorder,
        "comet": CometRecorder,
        "neptune": NeptuneRecorder,
    }
    unknown_options = set(backend_options) - recorder_types.keys()
    if unknown_options:
        raise ValueError(
            "recorder_kwargs only accepts tracking backends; unknown keys: "
            + ", ".join(sorted(unknown_options))
        )

    recorders = []
    for rec in config.recorders:
        if rec == "log":
            recorders.append(
                LogRecorder(log_path=output_dir / f"{exp_name}.log", console=True)
            )
        elif rec in recorder_types:
            # Aim needs a shared repository across runs for comparisons.
            path = Path(get_original_cwd()) / "aim" if rec == "aim" else output_dir
            kwargs = {
                "project": config.project,
                "name": exp_name,
                "group": group,
                "config": config_data,
                "tags": backend_tags,
                "path": path,
            }
            kwargs.update(backend_options.get(rec, {}))
            recorders.append(recorder_types[rec](**kwargs))
        else:
            raise ValueError(f"Unknown recorder: {rec}")
    return recorders
