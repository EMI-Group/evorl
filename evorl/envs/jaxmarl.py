import jax
import jax.numpy as jnp
import chex
import jaxmarl
from jaxmarl.environments import MultiAgentEnv

from evorl.types import (
    PyTreeDict, Action
)
from .multi_agent_env import MultiAgentEnvAdapter
from .env import EnvState
from .utils import sort_dict
from typing import Tuple
from evorl.utils.jax_utils import tree_zeros_like


def get_random_actions(batch_shape: Tuple[int], env: MultiAgentEnv):
    dummy_action_dict = {}
    for agent, action_space in zip(env.agents, env.action_space):
        dummy_action = env.action_space.sample()
        dummy_action_dict[agent] = jnp.broadcast_to(
            dummy_action, batch_shape+dummy_action.shape)

    return dummy_action_dict


class JaxMARLAdapter(MultiAgentEnvAdapter):
    def __init__(self, env: MultiAgentEnv):
        super(JaxMARLAdapter, self).__init__(env)

    def reset(self, key: chex.PRNGKey) -> EnvState:
        key, reset_key, dummy_step_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(reset_key)

        action = get_random_actions(obs.shape[:1], self.env)
        
        # run one dummy step to get reward,done,info shape
        _, _, dummy_reward, dummy_done, dummy_info = self.env.step_env(
            dummy_step_key, env_state, action)

        info = PyTreeDict(sort_dict(dummy_info))
        info.step_key = key 

        return EnvState(
            env_state=env_state,
            obs=obs,
            reward=tree_zeros_like(dummy_reward, dtype=jnp.float32),
            done=tree_zeros_like(dummy_done, dtype=jnp.float32),
            info=info
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        key, step_key = jax.random.split(state.info.step_key)

        # call step_env() instead of step() to disable autoreset
        # we handle the autoreset at AutoResetWrapper
        obs, env_state, reward, done, info = self.env.step_env(
            step_key, state.env_state, action)
        reward = reward.astype(jnp.float32)
        done = done.astype(jnp.float32)

        state.info.update(info)
        state.info.step_key = key

        return state.replace(
            env_state=env_state,
            obs=obs,
            reward=reward,
            done=done
        )

    @property
    def action_space(self):
        return self.env.action_spaces

    @property
    def obs_space(self):
        return self.env.observation_spaces


def create_jaxmarl_env(env_name: str, **kwargs) -> JaxMARLAdapter:
    env = jaxmarl.make(env_name, **kwargs)

    #TODO: add jaxmarl's vamp and log wrapper

    env = JaxMARLAdapter(env)

    return env
