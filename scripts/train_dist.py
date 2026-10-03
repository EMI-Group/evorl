import logging
import os

import hydra
from accelerator_utils import gpu_visibility_settings
from hydra.core.hydra_config import HydraConfig
from hydra_utils import (
    get_output_dir,
    set_absl_log_level,
    set_omegaconf_resolvers,
)
from omegaconf import DictConfig, OmegaConf
from recorder_setup import setup_recorders

logger = logging.getLogger("train_dist")

set_absl_log_level("warning")
set_omegaconf_resolvers()


def set_gpu_id():
    """Probe GPUs in a short subprocess, then select one before importing JAX."""
    job_id = HydraConfig.get().job.num
    settings, device, num_gpus = gpu_visibility_settings(job_id)
    if job_id >= num_gpus:
        logger.warning("It's not recommended to run multiple jobs on a single device.")
    logger.info(
        "Using %s device %s (%s)",
        settings["JAX_PLATFORMS"],
        device["local_hardware_id"],
        device["device_kind"],
    )
    os.environ.update(settings)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_dist(config: DictConfig) -> None:
    set_gpu_id()

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
        raise ValueError(
            f"In Parallel Training Mode, each job should only use one GPU/TPU, but find {devices}"
        )
    else:
        workflow: Workflow = workflow_cls.build_from_config(
            config, enable_jit=config.enable_jit
        )

    recorders = setup_recorders(config, workflow_cls.name(), parallel=True)
    workflow.add_recorders(recorders)

    try:
        state = workflow.init(jax.random.PRNGKey(config.seed))
        state = workflow.learn(state)
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise
    finally:
        workflow.close()


if __name__ == "__main__":
    train_dist()
