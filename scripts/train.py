import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra_utils import (
    get_output_dir,
    set_omegaconf_resolvers,
    set_absl_log_level,
)

logger = logging.getLogger("train")

set_absl_log_level("warning")
set_omegaconf_resolvers()


def setup_recorders(config: DictConfig, workflow_name: str):
    output_dir = Path(config.output_dir)

    from evorl.recorders import LogRecorder, WandbRecorder

    recorders = []
    tags = OmegaConf.to_container(config.tags, resolve=True)
    exp_name = "_".join([workflow_name, config.env.env_name, config.env.env_type])
    if len(tags) > 0:
        exp_name = exp_name + "|" + ",".join(tags)

    for rec in config.recorders:
        match rec:
            case "wandb":
                wandb_tags = [
                    workflow_name,
                    config.env.env_name,
                    config.env.env_type,
                ] + tags

                wandb_recorder = WandbRecorder(
                    project=config.project,
                    name=exp_name,
                    group="dev",
                    config=OmegaConf.to_container(
                        config, resolve=True
                    ),  # save the unrescaled config
                    tags=wandb_tags,
                    path=output_dir,
                )
                recorders.append(wandb_recorder)
            case "log":
                log_recorder = LogRecorder(
                    log_path=output_dir / f"{exp_name}.log", console=True
                )
                recorders.append(log_recorder)
            case _:
                raise ValueError(f"Unknown recorder: {rec}")

    return recorders

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(config: DictConfig) -> None:
    import jax
    from evorl.workflows import Workflow

    # jax.config.update("jax_threefry_partitionable", True)

    output_dir = get_output_dir()
    config.output_dir = str(output_dir)

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

    recorders = setup_recorders(config, workflow_cls.name())
    workflow.add_recorders(recorders)

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
