import logging
from hydra import compose, initialize
from pathlib import Path
import jax

from evorl.recorders import LogRecorder
from evorl.algorithms.ppo import PPOWorkflow
from evorl.utils.orbax_utils import save

# This ensures that log messages show up in the console.
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

output_dir = Path("./debug_temp")
if not output_dir.exists():
    output_dir.mkdir()
exp_name = "PPO_Hopper"

with initialize(version_base=None, config_path="configs"):
    # choose the config file:
    # env: configs/brax/hopper.yaml
    # algorithm: configs/agent/ppo.yaml
    config = compose(config_name="config", overrides=["env=brax/hopper", "agent=ppo"])

workflow = PPOWorkflow.build_from_config(config, enable_jit=config.enable_jit)

log_recorder = LogRecorder(log_path=output_dir / f"{exp_name}.log", console=True)
workflow.add_recorders([log_recorder])

state = workflow.init(jax.random.PRNGKey(config.seed))

# Train in one line:
# state = workflow.learn(state)

# Or manually control the training loop:
for i in range(100):
    train_metrics, state = workflow.step(state)
    if i % 10 == 0:
        eval_metrics, state = workflow.evaluate(state)
        workflow.recorder.write(eval_metrics.to_local_dict(), i)

        save(output_dir / f"state_{i}", state)

# release resources like checkpoint manager, recorders.
workflow.close()
