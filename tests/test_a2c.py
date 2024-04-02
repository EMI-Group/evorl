import jax
import jax.numpy as jnp
import chex
from evorl.agents.a2c import A2CWorkflow, A2CAgent, rollout

from hydra import compose, initialize
from evorl.envs import create_brax_env
from omegaconf import OmegaConf


def test_a2c():
    with initialize(config_path='../configs'):
        cfg = compose(config_name="config", overrides=["agent=a2c"])
    
    A2CWorkflow.enable_jit()
    learner = A2CWorkflow.build_from_config(cfg)
    state = learner.init(jax.random.PRNGKey(42))
    train_metric, state = learner.step(state)
    eval_metric, state = learner.evaluate(state)


def _create_example_agent_env(num_envs, rollout_length):
    config = OmegaConf.create()
    config.env = 'ant'
    config.num_envs = num_envs
    config.rollout_length = rollout
    env = create_brax_env(
        config.env, parallel=config.num_envs, autoreset=True)
    agent = A2CAgent(
        action_space=env.action_space,
        obs_space=env.obs_space,
        continuous_action=True,
    )

    return agent, env


def test_a2c_agent():
    agent, _ = _create_example_agent_env(4, 11)
    hash(agent)


def test_a2c_rollout():
    agent, env = _create_example_agent_env(4, 11)

    env_key, agent_key, rollout_key = jax.random.split(
        jax.random.PRNGKey(42), 3)

    env_state = env.reset(env_key)
    agent_state = agent.init(agent_key)

    env_nstate, trajectory = rollout(
        env,
        env_state,
        agent,
        agent_state,
        rollout_key,
        rollout_length=11,
        env_extra_fields=('last_obs',)
    )

    extras = trajectory.extras

    chex.assert_tree_shape_prefix(extras, (4, 11))
    assert 'raw_action' in extras['policy_extras']
    assert 'last_obs' in extras['env_extras']
