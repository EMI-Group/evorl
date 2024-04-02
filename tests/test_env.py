import jax
import chex
from evorl.agents.random_agent import RandomAgent
from evorl.envs import create_env

def _test_info_keys(env_state):
    for key in ('steps', 'termination', 'truncation', 'last_obs', 'episode_return', 'reset_key'):
        assert key in env_state.info, f'Missing key {key} in env_state.info'


def test_brax():
    num_envs = 7
    env = create_env(
        'ant',
        'brax',
        parallel=num_envs,
        autoreset=True,
        discount=0.99
    )

    agent = RandomAgent(
        action_space=env.action_space,
        obs_space=env.obs_space
    )

    env_key, agent_key, step_key = jax.random.split(jax.random.PRNGKey(42), 3)

    env_state = env.reset(env_key)
    chex.assert_shape(env_state.obs, (num_envs, *env.obs_space.shape))
    _test_info_keys(env_state)

    agent_state = agent.init(agent_key)

    for i in range(15):
        action, _ = agent.compute_actions(agent_state, env_state, step_key)
        env_state = env.step(env_state, action)

    _test_info_keys(env_state)

