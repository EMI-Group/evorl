import math
import jax
import jax.numpy as jnp


from evorl.rollout import rollout
from evorl.envs import (
    create_brax_env,
    Box,
)
from evorl.envs.wrappers import (
    OneEpisodeWrapper,
    RewardScaleWrapper,
    ActionRepeatWrapper,
    VmapWrapper,
)
from evorl.utils.rl_toolkits import compute_discount_return

from .utils import DebugRandomAgent, FakeEnv


def test_reward_scale_wrapper():
    reward_scale = 7.0

    env = create_brax_env("ant")
    env = RewardScaleWrapper(env, reward_scale=reward_scale)
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
    episode_return = trajectory.extras.env_extras.episode_return[-1, :]
    episode_return2 = compute_discount_return(reward, trajectory.dones)

    assert jnp.allclose(reward, ori_reward * reward_scale)
    assert jnp.allclose(episode_return, episode_return2)


def test_action_repeat_wrapper():
    rollout_length = 17
    action_repeat = 4
    reward_scale = 7.0
    new_rollout_length = math.ceil(rollout_length / action_repeat)

    rewards = jnp.arange(1, rollout_length + 1, dtype=jnp.float32)
    dones = jnp.zeros((rollout_length,), dtype=jnp.float32)
    dones = dones.at[-1].set(1.0)

    env = FakeEnv(rewards, dones)
    env = RewardScaleWrapper(env, reward_scale=reward_scale)
    env = OneEpisodeWrapper(env, episode_length=rollout_length, discount=1.0)
    env = ActionRepeatWrapper(env, action_repeat=action_repeat)
    env = VmapWrapper(env, num_envs=1)
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
        rollout_length=new_rollout_length,
        env_extra_fields=("ori_reward", "steps", "episode_return"),
    )

    assert trajectory.dones.shape[0] == new_rollout_length

    ori_rewards = trajectory.extras.env_extras.ori_reward

    real_acc_rewards = jnp.stack(
        [
            rewards[i : i + action_repeat].sum(keepdims=True)
            for i in range(0, len(rewards), action_repeat)
        ]
    )  # [T//action_repeat, 1]
    assert jnp.allclose(ori_rewards, real_acc_rewards, atol=1e-5, rtol=0)
    assert jnp.allclose(
        trajectory.rewards, real_acc_rewards * reward_scale, atol=1e-5, rtol=0
    )
