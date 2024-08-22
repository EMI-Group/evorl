import os
import jax
import jax.numpy as jnp
import chex

from evorl.envs import Env, Box, Discrete, EnvState
from evorl.types import PyTreeDict


def disable_gpu_preallocation():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def enable_nan_inf_check():
    os.environ["JAX_DEBUG_NANS"] = "true"


def enable_deterministic_mode():
    xla_flags = os.getenv("XLA_FLAGS", "")
    # print(f"current XLA_FLAGS: {xla_flags}")
    if len(xla_flags) > 0:
        xla_flags = xla_flags + " "
    os.environ["XLA_FLAGS"] = xla_flags + "--xla_gpu_deterministic_ops=true"


class FakeVmapEnv(Env):
    def __init__(self, rewards, dones):
        chex.assert_equal_shape([rewards, dones])
        self._rewards = rewards  # [T,B]
        self._dones = dones
        self.max_episode_length = rewards.shape[0]
        self.num_envs = rewards.shape[1]

    def reset(self, key):
        if key.ndim > 1:
            key = key[0]

        return EnvState(
            env_state=PyTreeDict(
                i=jnp.zeros((), dtype=jnp.int32),
                key=key,
            ),
            obs=self._create_obs(key),
            reward=jnp.zeros(self.num_envs),
            done=jnp.zeros(self.num_envs),
        )

    def step(self, state, action):
        i = state.env_state.i
        key = state.env_state.key
        reward = self._rewards[i]
        done = self._dones[i]
        i = i + 1

        env_state = state.env_state.replace(i=i, key=key)
        obs = self._create_obs(key)
        return state.replace(env_state=env_state, obs=obs, reward=reward, done=done)

    def _create_obs(self, key):
        obs = self.obs_space.sample(key)
        obs = jnp.broadcast_to(obs, (self.num_envs,) + obs.shape)
        return obs

    @property
    def obs_space(self):
        return Box(low=-jnp.ones(7), high=jnp.ones(7))

    @property
    def action_space(self):
        return Discrete(5)
