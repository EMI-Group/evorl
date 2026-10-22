"""Small vectorized environments for workflow regression tests."""

import jax
import jax.numpy as jnp
from evorl.envs import Box, Discrete, Env, EnvState
from evorl.types import PyTreeDict


class ToyEnv(Env):
    def __init__(self, num_envs, discrete=False):
        self.num_envs = num_envs
        self.discrete = discrete

    @property
    def obs_space(self):
        return Box(low=-jnp.ones(3), high=jnp.ones(3))

    @property
    def action_space(self):
        if self.discrete:
            return Discrete(n=2)
        return Box(low=-jnp.ones(1), high=jnp.ones(1))

    def reset(self, key):
        obs = jax.random.uniform(key, (self.num_envs, 3))
        zeros = jnp.zeros(self.num_envs)
        return EnvState(
            env_state=jnp.zeros(self.num_envs, dtype=jnp.int32),
            obs=obs,
            reward=zeros,
            done=zeros,
            info=PyTreeDict(
                autoreset=zeros, episode_return=zeros, termination=zeros, ori_obs=obs
            ),
        )

    def step(self, state, action):
        count = state.env_state + 1
        done = (count % 2 == 0).astype(jnp.float32)
        reward = 1 - (
            action.astype(jnp.float32) if self.discrete else jnp.square(action).sum(-1)
        )
        obs = state.obs + 0.1
        return state.replace(
            env_state=count,
            obs=obs,
            reward=reward,
            done=done,
            info=state.info.replace(
                autoreset=jnp.zeros_like(done),
                episode_return=2 * reward,
                termination=done,
                ori_obs=obs,
            ),
        )


class VectorRewardEnv(ToyEnv):
    def __init__(self, num_envs, num_objectives=2):
        super().__init__(num_envs)
        self.num_objectives = num_objectives

    def reset(self, key):
        state = super().reset(key)
        return state.replace(reward=jnp.zeros((self.num_envs, self.num_objectives)))

    def step(self, state, action):
        state = super().step(state, action)
        rewards = [
            1 - jnp.square(action - 0.5 * i).sum(-1) for i in range(self.num_objectives)
        ]
        return state.replace(reward=jnp.stack(rewards, axis=-1))
