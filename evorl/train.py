import jax

from omegaconf import DictConfig, OmegaConf
import hydra
import logging

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

    state = workflow.init(jax.random.PRNGKey(config.seed))
    state = workflow.learn(state)


if __name__ == "__main__":
    train()