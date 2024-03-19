import jax
import jax.numpy as jnp
import chex
from evorl.agents.a2c import A2CWorkflow, A2CAgent, rollout

from hydra import compose, initialize
from evorl.envs import create_brax_env
from omegaconf import OmegaConf


def test_a2c():
    with initialize(config_path='../configs'):
        cfg = compose(config_name='a2c')
    learner = A2CWorkflow(cfg)
    state = learner.init(jax.random.PRNGKey(42))
    nstate = learner.step(state)


def test_a2c_rollout():
    config = OmegaConf.creat()

    config.env = 'ant'
    config.num_envs = 4
    config.rollout_length = 11

    env = create_brax_env(
        config.env, parallel=config.num_envs, autoset=True)

    agent = A2CAgent(
        action_space=env.action_space,
        obs_space=env.obs_space,
        continuous_action=True,
        gae_lambda=config.gae_lambda,
        discount=config.discount
    )

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
        rollout_length=config.rollout_length,
        extra_fields=('last_obs',)
    )

    extras = trajectory.extras

    chex.assert_tree_shape_prefix(extras, (4, 11))
    assert 'raw_action' in extras['policy_extras']
    assert 'last_obs' in extras['env_extras']
