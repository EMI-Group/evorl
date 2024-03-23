import jax
import chex

from evorl.rollout import rollout, rollout_episode, rollout_episode_mod, eval_rollout, eval_rollout_episode

from evorl.agents.random_agent import RandomAgent
from evorl.envs import create_brax_env


def test_rollout():
    env = create_brax_env(
        "ant",
        parallel=7,
        autoreset=True
    )

    agent= RandomAgent(
        action_space=env.action_space,
        obs_space=env.obs_space
    )

    key = jax.random.PRNGKey(42)

    rollout_key, env_key, agent_key = jax.random.split(key,3)

    env_state = env.reset(env_key)

    agent_state = agent.init(agent_key)

    env_nstate, trajectory = rollout(
        env,
        agent,
        env_state,
        agent_state,
        rollout_key,
        rollout_length=1000,
        env_extra_fields=('termination','truncation')
    )





def test_rollout():
    env = create_brax_env(
        "ant",
        parallel=6,
        autoreset=False
    )

    agent= RandomAgent(
        action_space=env.action_space,
        obs_space=env.obs_space
    )

    key = jax.random.PRNGKey(42)

    rollout_key, env_key, agent_key = jax.random.split(key,3)

    env_state = env.reset(env_key)

    agent_state = agent.init(agent_key)

    env_nstate, episode = rollout_episode_mod(
        env,
        agent,
        env_state,
        agent_state,
        rollout_key,
        rollout_length=1000,
        env_extra_fields=('termination','truncation')
    )

    valid_mask = episode.valid_mask
    trajectory = episode.trajectory


