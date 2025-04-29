import os
import jax.numpy as jnp
import jax.tree_util as jtu
import chex

from evorl.agent import Agent, AgentState
from evorl.envs import Env, Box, Discrete, EnvState, Space
from evorl.sample_batch import SampleBatch
from evorl.types import PolicyExtraInfo, PyTreeDict, Action
from evorl.utils.jax_utils import tree_get


def set_default_device_cpu():
    os.environ["JAX_PLATFORM_NAME"] = "cpu"


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


class FakeEnv(Env):
    def __init__(self, rewards, dones):
        chex.assert_equal_shape([rewards, dones])
        self._rewards = rewards
        self._dones = dones # [T]
        self.max_episode_length = dones.shape[0]

    def reset(self, key):
        return EnvState(
            env_state=PyTreeDict(
                i=jnp.zeros((), dtype=jnp.int32),
                key=key,
            ),
            obs=self._create_obs(key),
            reward=jtu.tree_map(lambda x: jnp.zeros(()), self._rewards),
            done=jnp.zeros(()),
        )

    def step(self, state, action):
        i = state.env_state.i
        key = state.env_state.key
        reward = self._rewards[i]
        done = self._dones[i]

        env_state = state.env_state.replace(i=i + 1, key=key)
        obs = self._create_obs(key)
        return state.replace(env_state=env_state, obs=obs, reward=reward, done=done)

    def _create_obs(self, key):
        return self.obs_space.sample(key)

    @property
    def obs_space(self):
        return Box(low=-jnp.ones(7), high=jnp.ones(7))

    @property
    def action_space(self):
        return Discrete(n=5)


class FakeVmapEnv(Env):
    def __init__(self, rewards, dones):
        chex.assert_equal_shape([rewards, dones])
        self._rewards = rewards  # [T,B]
        self._dones = dones
        self.max_episode_length = dones.shape[0]
        self.num_envs = dones.shape[1]

    def reset(self, key):
        if key.ndim > 1:
            key = key[0]

        return EnvState(
            env_state=PyTreeDict(
                i=jnp.zeros((), dtype=jnp.int32),
                key=key,
            ),
            obs=self._create_obs(key),
            reward=jtu.tree_map(lambda x: jnp.zeros(self.num_envs), self._rewards),
            done=jnp.zeros(self.num_envs),
        )

    def step(self, state, action):
        i = state.env_state.i
        key = state.env_state.key
        reward = tree_get(self._rewards, i)
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
        return Discrete(n=5)


class DebugRandomAgent(Agent):
    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        extra_state = PyTreeDict(
            action_space=action_space,
            obs_space=obs_space,
        )
        return AgentState(params={}, extra_state=extra_state)

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs_space = agent_state.extra_state.obs_space
        action_space = agent_state.extra_state.action_space

        batch_shapes = sample_batch.obs.shape[: -len(obs_space.shape)]
        actions = action_space.sample(key)
        actions = jnp.broadcast_to(actions, batch_shapes + actions.shape)
        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        return self.compute_actions(agent_state, sample_batch, key)
