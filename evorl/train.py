import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from evorl.utils.hydra_utils import (
    get_output_dir,
    set_omegaconf_resolvers,
    set_absl_log_level,
)

logger = logging.getLogger("train")

set_absl_log_level("warning")
set_omegaconf_resolvers()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(config: DictConfig) -> None:
    import jax
    from evorl.recorders import LogRecorder, WandbRecorder
    from evorl.workflows import Workflow

    jax.config.update("jax_threefry_partitionable", True)

    logger.info("config:\n" + OmegaConf.to_yaml(config, resolve=True))

    workflow_cls = hydra.utils.get_class(config.workflow_cls)
    workflow_cls = type(workflow_cls.__name__, (workflow_cls,), {})

    devices = jax.local_devices()
    if len(devices) > 1:
        logger.info(f"Enable Multiple Devices: {devices}")
        workflow: Workflow = workflow_cls.build_from_config(
            config, enable_multi_devices=True
        )
    else:
        workflow: Workflow = workflow_cls.build_from_config(
            config, enable_jit=config.enable_jit
        )

    output_dir = get_output_dir()
    wandb_project = config.wandb.project
    cfg_wandb_tags = OmegaConf.to_container(config.wandb.tags, resolve=True)
    wandb_tags = [
        workflow_cls.name(),
        config.env.env_name,
        config.env.env_type,
    ] + cfg_wandb_tags
    wandb_name = "_".join(
        [workflow_cls.name(), config.env.env_name, config.env.env_type]
    )
    if len(cfg_wandb_tags) > 0:
        wandb_name = wandb_name + "|" + ",".join(cfg_wandb_tags)
    wandb_mode = None if config.wandb.enable and not config.debug else "disabled"

    wandb_recorder = WandbRecorder(
        project=wandb_project,
        name=wandb_name,
        group="dev",
        config=OmegaConf.to_container(
            config, resolve=True
        ),  # save the unrescaled config
        tags=wandb_tags,
        path=output_dir,
        mode=wandb_mode,
    )
    log_recorder = LogRecorder(log_path=output_dir / f"{wandb_name}.log", console=True)
    workflow.add_recorders([wandb_recorder, log_recorder])

    try:
        state = workflow.init(jax.random.PRNGKey(config.seed))
        state = workflow.learn(state)
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise e
    finally:
        workflow.close()


if __name__ == "__main__":
    train()
