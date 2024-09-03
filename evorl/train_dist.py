import logging

import os
import hydra
import jax
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from evorl.recorders import LogRecorder, WandbRecorder
from evorl.utils.cfg_utils import get_output_dir, set_omegaconf_resolvers
from evorl.utils.jax_utils import optimize_gpu_utilization
from evorl.workflows import Workflow

logger = logging.getLogger("train")

optimize_gpu_utilization()
jax.config.update("jax_compilation_cache_dir", "../jax-cache")
jax.config.update("jax_threefry_partitionable", True)
set_omegaconf_resolvers()


def set_gpu_id():
    if HydraConfig.initialized():
        job_id = HydraConfig.get().job.num
        num_gpus = jax.device_count()
        gpu_id = job_id % num_gpus
    else:
        gpu_id = 0

    logger.info(f"Using GPU {gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_dist(config: DictConfig) -> None:
    set_gpu_id()

    logger.info("config:\n" + OmegaConf.to_yaml(config))

    if config.debug:
        from jax import config as jax_config

        jax_config.update("jax_debug_nans", True)
        # jax.config.update("jax_transfer_guard", "log")
        # jax_config.update("jax_debug_infs", True)

    workflow_cls = hydra.utils.get_class(config.workflow_cls)
    workflow_cls = type(workflow_cls.__name__, (workflow_cls,), {})

    from evorl.utils.jax_utils import is_jitted

    print(f"Jitted step: {is_jitted(workflow_cls.step)}")

    devices = jax.local_devices()
    if len(devices) > 1:
        raise ValueError(
            f"In Parallel Training Mode, each job should only use one GPU/TPU, but find {devices}"
        )
    else:
        workflow: Workflow = workflow_cls.build_from_config(
            config, enable_jit=config.enable_jit
        )

    output_dir = get_output_dir()
    wandb_project = config.wandb.project
    cfg_wandb_tags = OmegaConf.to_container(config.wandb.tags)
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
        group=wandb_name,
        config=OmegaConf.to_container(config),  # save the unrescaled config
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
    train_dist()
