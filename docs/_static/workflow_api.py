from hydra import compose, initialize
import jax

from evorl.algorithms.ppo import PPOWorkflow


with initialize(version_base=None, config_path="configs"):
    # choose the config file:
    # env: configs/brax/hopper.yaml
    # algorithm: configs/agent/ppo.yaml
    config = compose(config_name="config", overrides=["env=brax/hopper", "agent=ppo"])

workflow = PPOWorkflow.build_from_config(config, enable_jit=config.enable_jit)
state = workflow.init(jax.random.PRNGKey(config.seed))

# Train in one line:
# state = workflow.learn(state)

# Or manually control the training loop:
for i in range(100):
    train_metrics, state = workflow.step(state)
    if i % 10 == 0:
        eval_metrics, state = workflow.evaluate(state)
        print(
            f"Step {i} Avg Return: {eval_metrics.episode_returns} Avg Length: {eval_metrics.episode_lengths}"
        )

# release resources like checkpoint manager, recorders.
workflow.close()
