import jax
import jax.numpy as jnp

from evorl.rollout import rollout
from evorl.envs import (
    create_brax_env,
    Box,
)
from evorl.envs.wrappers import OneEpisodeWrapper, RewardScaleWrapper, VmapWrapper
from evorl.utils.rl_toolkits import compute_discount_return

from .utils import DebugRandomAgent


def test_reward_scale_wrapper():
    scale = 7.0

    env = create_brax_env("ant")
    env = RewardScaleWrapper(env, scale=scale)
    env = OneEpisodeWrapper(env, episode_length=1000, discount=1.0)
    env = VmapWrapper(env, num_envs=3)
    agent = DebugRandomAgent()

    key = jax.random.PRNGKey(42)
    rollout_key, env_key, agent_key = jax.random.split(key, 3)

    env_state = env.reset(env_key)
    agent_state = agent.init(env.obs_space, env.action_space, agent_key)

    trajectory, env_nstate = rollout(
        env.step,
        agent.compute_actions,
        env_state,
        agent_state,
        rollout_key,
        rollout_length=1000,
        env_extra_fields=("ori_reward", "steps", "episode_return"),
    )

    reward = trajectory.rewards
    ori_reward = trajectory.extras.env_extras.ori_reward

    # [T,B] -> [B]
    episode_return = trajectory.extras.env_extras.episode_return[-1,:]
    episode_return2 = compute_discount_return(
        reward, trajectory.dones
    )

    assert jnp.allclose(reward, ori_reward*scale)
    assert jnp.allclose(episode_return, episode_return2)
