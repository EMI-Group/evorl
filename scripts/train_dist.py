import os
import logging
import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra_utils import (
    get_output_dir,
    set_omegaconf_resolvers,
    set_absl_log_level,
)

logger = logging.getLogger("train_dist")

set_absl_log_level("warning")
set_omegaconf_resolvers()

"""
Note: this script currently only support Nvidia GPUs.
"""


def get_gpus_info():
    # Run the nvidia-smi command to list GPUs and count the lines
    output = subprocess.check_output("nvidia-smi --list-gpus", shell=True)
    # Decode the output from bytes to a string and count lines
    return output.decode().splitlines()


def set_gpu_id():
    gpus_info = get_gpus_info()

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    if cuda_visible_devices != "all" and len(cuda_visible_devices) > 0:
        gpu_ids = [int(i) for i in cuda_visible_devices.split(",")]
        num_gpus = len(gpu_ids)
    else:
        num_gpus = len(gpus_info)
        gpu_ids = list(range(num_gpus))

    job_id = HydraConfig.get().job.num
    gpu_idx = job_id % num_gpus

    if job_id >= num_gpus:
        logger.warning("It's not recommended to run multiple jobs on a single device.")

    gpu_id = gpu_ids[gpu_idx]

    logger.info(f"Using {gpus_info[gpu_id]}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_dist(config: DictConfig) -> None:
    set_gpu_id()

    import jax
    from evorl.recorders import LogRecorder, WandbRecorder
    from evorl.workflows import Workflow

    jax.config.update("jax_threefry_partitionable", True)

    output_dir = get_output_dir()
    config.output_dir = str(output_dir)

    logger.info("config:\n" + OmegaConf.to_yaml(config, resolve=True))

    workflow_cls = hydra.utils.get_class(config.workflow_cls)
    workflow_cls = type(workflow_cls.__name__, (workflow_cls,), {})

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
    tags = OmegaConf.to_container(config.tags, resolve=True)
    wandb_tags = [
        workflow_cls.name(),
        config.env.env_name,
        config.env.env_type,
    ] + tags
    wandb_name = "_".join(
        [workflow_cls.name(), config.env.env_name, config.env.env_type]
    )
    if len(tags) > 0:
        wandb_name = wandb_name + "|" + ",".join(tags)

    wandb_recorder = WandbRecorder(
        project=config.project,
        name=wandb_name,
        group=wandb_name,
        config=OmegaConf.to_container(
            config, resolve=True
        ),  # save the unrescaled config
        tags=wandb_tags,
        path=output_dir,
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
