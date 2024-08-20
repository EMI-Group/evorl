import jax
import jax.numpy as jnp
import chex
from evorl.agents.a2c import A2CWorkflow, A2CAgent, rollout

from hydra import compose, initialize
from evorl.envs import create_wrapped_brax_env
from omegaconf import OmegaConf


def setup_a2c():
    with initialize(config_path="../configs"):
        cfg = compose(config_name="config", overrides=["agent=a2c", "env=brax/ant"])

    workflow = A2CWorkflow.build_from_config(cfg, enable_jit=True)

    return workflow


def test_a2c():
    workflow = setup_a2c()
    state = workflow.init(jax.random.PRNGKey(42))
    train_metric, state = workflow.step(state)
    eval_metric, state = workflow.evaluate(state)


def test_a2c_learn():
    workflow = setup_a2c()
    state = workflow.init(jax.random.PRNGKey(42))
    state = workflow.learn(state)


def _create_example_agent_env(num_envs, rollout_length):
    env = "ant"
    num_envs = num_envs

    env = create_wrapped_brax_env(env, parallel=num_envs)
    agent = A2CAgent(
        continuous_action=True,
    )

    return agent, env


def test_agent_hashable():
    agent, env = _create_example_agent_env(5, 1000)
    agent.init(env.obs_space, env.action_space, jax.random.PRNGKey(42))
    hash(agent)
