import jax
from omegaconf import DictConfig, OmegaConf
import hydra
import orbax.checkpoint as ocp

from evorl.utils.cfg_utils import get_output_dir
from evorl.recorders import WandbRecorder, LogRecorder, ChainRecorder
from evorl.distributed import tree_unpmap
from pathlib import Path
import logging

from evorl.agents.ddpg import DDPGAgent, DDPGWorkflow

logger = logging.getLogger(__name__)
path = "./outputs/train/2024-04-12_09-27-58"
last_step = 97700
# Assume the checkpoint manager setup function and any necessary imports are available
def setup_checkpoint_manager(config: DictConfig) -> ocp.CheckpointManager:
    # output_dir = get_output_dir()
    output_dir = path
    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=config.checkpoint.save_interval_steps,
        max_to_keep=config.checkpoint.max_to_keep
    )
    ckpt_path = output_dir + '/checkpoints'
    logger.info(f'Set checkpoint path: {ckpt_path}')
    checkpoint_manager = ocp.CheckpointManager(
        ckpt_path,
        options=ckpt_options,
        metadata=OmegaConf.to_container(config)  # Rescaled real config
    )
    return checkpoint_manager

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def load_model(config: DictConfig) -> None:
    logger.info("Config:\n" + OmegaConf.to_yaml(config))
    if config.debug:
        from jax import config as jax_config
        jax_config.update("jax_debug_nans", True)
    workflow_cls = hydra.utils.get_class(config.workflow_cls)

    # devices = jax.local_devices()
    
    workflow = workflow_cls.build_from_config(
            config,
            enable_jit=False
        )
    
    output_dir = get_output_dir()
    wandb_project = config.wandb.project
    wandb_tags = [workflow_cls.name(), config.env.env_name, config.env.env_type] + \
        OmegaConf.to_container(config.wandb.tags)
    wandb_name = '-'.join(
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
    log_recorder = LogRecorder(log_path=output_dir/f'{wandb_name}.log')
    workflow.add_recorders([wandb_recorder, log_recorder])
    # Setup the checkpoint manager
    checkpoint_manager = setup_checkpoint_manager(config)
    
    state = workflow.init(jax.random.PRNGKey(config.seed))
    pmap_axis_name = None
    # state = checkpoint_manager.restore(last_step)
    # target_state = {'layer0': {'bias': 0.0, 'weight': 0.0}}
    # state = checkpoint_manager.restore(last_step, args=ocp.args.StandardRestore(tree_unpmap(state,None)))
    state = checkpoint_manager.restore(last_step, args=ocp.args.StandardRestore(state))
    print("finished loading model")

if __name__ == "__main__":
    load_model()
