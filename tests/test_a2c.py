import jax
from evorl.algorithms.a2c import A2CWorkflow, make_mlp_a2c_agent

from hydra import compose, initialize
from evorl.envs import create_wrapped_brax_env


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


def _create_example_agent_env(num_envs):
    env = "ant"
    num_envs = num_envs

    env = create_wrapped_brax_env(env, parallel=num_envs)
    agent = make_mlp_a2c_agent(env.action_space)

    return agent, env


def test_agent():
    agent, env = _create_example_agent_env(5)

    # test init
    agent.init(env.obs_space, env.action_space, jax.random.PRNGKey(42))

    # test hashable
    hash(agent)
