import jax

from omegaconf import DictConfig, OmegaConf
import hydra
import logging

from hydra.core.hydra_config import HydraConfig
from evorl.recorders import WandbRecorder, LogRecorder, ChainRecorder
from pathlib import Path

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(config: DictConfig) -> None:
    logger.info("config:\n"+OmegaConf.to_yaml(config))

    workflow_cls = hydra.utils.get_class(config.workflow_cls)

    devices = jax.local_devices()
    if len(devices) > 1:
        logger.info(f"Enable Multi Devices: {devices}")
        workflow = workflow_cls.build_from_config(
            config, enable_multi_devices=True, devices=devices
        )
    else:
        workflow_cls.enable_jit()
        workflow = workflow_cls.build_from_config(config)


    output_dir = Path(HydraConfig.get().run.dir).absolute()
    wandb_project = config.wandb.project
    if config.debug:
        wandb_project = wandb_project + '_debug'
    wandb_tags = [workflow_cls.name(), config.env.env_name, config.env.env_type] + \
        OmegaConf.to_container(config.wandb.tags)
    wandb_name = ','.join(
        [workflow_cls.name(), config.env.env_name, config.env.env_type]
    )
    wandb_mode = 'online' if config.wandb.enable and not config.debug else 'disabled'

    wandb_recorder = WandbRecorder(
        project=wandb_project,
        name=wandb_name,
        config=OmegaConf.to_container(config),  # save the unrescaled config
        tags=wandb_tags,
        dir=output_dir,
        mode=wandb_mode
    )
    log_recorder = LogRecorder(log_path=output_dir/'train.log', console=config.debug)
    workflow.add_recorders([wandb_recorder, log_recorder])

    state = workflow.init(jax.random.PRNGKey(config.seed))
    state = workflow.learn(state)

    workflow.recorder.close()


if __name__ == "__main__":
    train()
