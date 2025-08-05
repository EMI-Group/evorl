import math
import jax
import jax.numpy as jnp

from evorl.envs import create_brax_env
from evorl.envs.wrappers import (
    OneEpisodeWrapper,
    RewardScaleWrapper,
    ActionRepeatWrapper,
    SparseRewardWrapper,
    VmapWrapper,
)
from evorl.rollout import rollout
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

    rewards = trajectory.rewards
    ori_rewards = trajectory.extras.env_extras.ori_reward

    # [T,B] -> [B]
    episode_return = trajectory.extras.env_extras.episode_return[-1, :]
    episode_return2 = compute_discount_return(rewards, trajectory.dones)

    assert jnp.allclose(rewards, ori_rewards * reward_scale)
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

    episode_return = trajectory.extras.env_extras.episode_return[-1, :]
    episode_return2 = (rewards * reward_scale).sum()
    assert jnp.allclose(episode_return, episode_return2, atol=1e-5, rtol=0)


def test_action_repeat_wrapper2():
    rollout_length = 1000
    action_repeat = 7
    reward_scale = 7.0
    new_rollout_length = math.ceil(rollout_length / action_repeat)

    env = create_brax_env("ant")
    env = RewardScaleWrapper(env, reward_scale=reward_scale)
    env = OneEpisodeWrapper(env, episode_length=rollout_length, discount=1.0)
    env = ActionRepeatWrapper(env, action_repeat=action_repeat)
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
        rollout_length=new_rollout_length,
        env_extra_fields=("ori_reward", "steps", "episode_return"),
    )

    rewards = trajectory.rewards
    ori_rewards = trajectory.extras.env_extras.ori_reward
    episode_return = trajectory.extras.env_extras.episode_return[-1, :]
    episode_return2 = compute_discount_return(rewards, trajectory.dones)

    assert jnp.allclose(rewards, ori_rewards * reward_scale)
    assert jnp.allclose(rewards.sum(axis=0), episode_return)
    assert jnp.allclose(episode_return, episode_return2)


def test_sparse_reward_wrapper():
    # sparse_length = 7
    # for rollout_length in range(21,29):
    sparse_length = 4
    for rollout_length in range(16, 21):
        rewards = jnp.arange(1, rollout_length + 1, dtype=jnp.float32)
        dones = jnp.zeros((rollout_length,), dtype=jnp.float32)
        dones = dones.at[-1].set(1.0)

        env = FakeEnv(rewards, dones)
        env = SparseRewardWrapper(env, sparse_length=sparse_length)
        env = OneEpisodeWrapper(env, episode_length=rollout_length, discount=1.0)
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
            rollout_length=rollout_length,
            env_extra_fields=("ori_reward", "steps", "episode_return"),
        )

        real_rewards = jnp.zeros_like(rewards)
        for i in range(0, len(rewards) + 1, sparse_length):
            real_rewards = real_rewards.at[i + sparse_length - 1].set(
                rewards[i : i + sparse_length].sum()
            )

        if i != len(rewards):
            real_rewards = real_rewards.at[-1].set(rewards[i:].sum())

        assert jnp.allclose(trajectory.rewards.reshape(-1), real_rewards)


def test_sparse_reward_wrapper2():
    rollout_length = 33
    sparse_length = 7
    term_idx = 30

    rewards = jnp.arange(1, rollout_length + 1, dtype=jnp.float32)
    dones = jnp.zeros((rollout_length,), dtype=jnp.float32)
    dones = dones.at[term_idx:].set(1.0)

    env = FakeEnv(rewards, dones)
    env = SparseRewardWrapper(env, sparse_length=sparse_length)
    env = OneEpisodeWrapper(env, episode_length=rollout_length, discount=1.0)
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
        rollout_length=rollout_length,
        env_extra_fields=("ori_reward", "steps", "episode_return"),
    )

    base = (term_idx // sparse_length) * sparse_length

    last_reward = rewards[base : term_idx + 1].sum()

    for r in trajectory.rewards[term_idx:].reshape(-1):
        assert jnp.allclose(last_reward, r)
