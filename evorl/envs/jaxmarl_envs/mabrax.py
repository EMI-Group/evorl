from functools import partial
from typing import Literal

import chex
import jax
import jax.numpy as jnp
from brax import envs
from gymnax.environments import spaces
from jaxmarl.environments.mabrax.mabrax_env import (
    MABraxEnv,
    _agent_action_mapping,
    _agent_observation_mapping,
)

# TODO: move homogenisation to a separate wrapper


class MABraxEnvV2(MABraxEnv):
    def __init__(
        self,
        env_name: str,
        homogenisation_method: Literal["max", "concat"] | None = None,
        backend: str = "positional",
        **kwargs,
    ):
        """Multi-Agent Brax environment.

        Compared to the original MABraxEnv, this version disables the autoreset and other wrappers in internal brax envs.

        Args:
            env_name: Name of the environment to be used.
            episode_length: Length of an episode. Defaults to 1000.
            action_repeat: How many repeated actions to take per environment
                step. Defaults to 1.
            auto_reset: Whether to automatically reset the environment when
                an episode ends. Defaults to True.
            homogenisation_method: Method to homogenise observations and actions
                across agents. If None, no homogenisation is performed, and
                observations and actions are returned as is. If "max", observations
                and actions are homogenised by taking the maximum dimension across
                all agents and zero-padding the rest. In this case, the index of the
                agent is prepended to the observation as a one-hot vector. If "concat",
                observations and actions are homogenised by masking the dimensions of
                the other agents with zeros in the full observation and action vectors.
                Defaults to None.
        """
        base_env_name = env_name.split("_")[0]
        env = envs.get_environment(base_env_name, backend=backend, **kwargs)

        self.env = env
        self.homogenisation_method = homogenisation_method
        self.agent_obs_mapping = _agent_observation_mapping[env_name]
        self.agent_action_mapping = _agent_action_mapping[env_name]
        self.agents = list(self.agent_obs_mapping.keys())

        self.num_agents = len(self.agent_obs_mapping)
        obs_sizes = {
            agent: (
                self.num_agents + max([o.size for o in self.agent_obs_mapping.values()])
                if homogenisation_method == "max"
                else (
                    self.env.observation_size
                    if homogenisation_method == "concat"
                    else obs.size
                )
            )
            for agent, obs in self.agent_obs_mapping.items()
        }
        act_sizes = {
            agent: (
                max([a.size for a in self.agent_action_mapping.values()])
                if homogenisation_method == "max"
                else (
                    self.env.action_size
                    if homogenisation_method == "concat"
                    else act.size
                )
            )
            for agent, act in self.agent_action_mapping.items()
        }

        self.observation_spaces = {
            agent: spaces.Box(
                -jnp.inf,
                jnp.inf,
                shape=(obs_sizes[agent],),
            )
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(
                -1.0,
                1.0,
                shape=(act_sizes[agent],),
            )
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def global_step_env(
        self,
        key: chex.PRNGKey,
        state: envs.State,
        global_action: chex.Array,
    ) -> tuple[
        dict[str, chex.Array], envs.State, dict[str, float], dict[str, bool], dict
    ]:
        next_state = self.env.step(state, global_action)
        observations = self.get_obs(next_state)
        rewards = {agent: next_state.reward for agent in self.agents}
        rewards["__all__"] = next_state.reward
        dones = {agent: next_state.done.astype(jnp.bool_) for agent in self.agents}
        dones["__all__"] = next_state.done.astype(jnp.bool_)
        return (
            observations,
            next_state,
            rewards,
            dones,
            next_state.info,
        )


class Ant(MABraxEnvV2):
    def __init__(self, **kwargs):
        super().__init__("ant_4x2", **kwargs)


class HalfCheetah(MABraxEnvV2):
    def __init__(self, **kwargs):
        super().__init__("halfcheetah_6x1", **kwargs)


class Hopper(MABraxEnvV2):
    def __init__(self, **kwargs):
        super().__init__("hopper_3x1", **kwargs)


class Humanoid(MABraxEnvV2):
    def __init__(self, **kwargs):
        super().__init__("humanoid_9|8", **kwargs)


class Walker2d(MABraxEnvV2):
    def __init__(self, **kwargs):
        super().__init__("walker2d_2x3", **kwargs)


def make_mabrax_env(env_id: str, **env_kwargs):
    if env_id == "ant_4x2":
        env = Ant(**env_kwargs)
    elif env_id == "halfcheetah_6x1":
        env = HalfCheetah(**env_kwargs)
    elif env_id == "hopper_3x1":
        env = Hopper(**env_kwargs)
    elif env_id == "humanoid_9|8":
        env = Humanoid(**env_kwargs)
    elif env_id == "walker2d_2x3":
        env = Walker2d(**env_kwargs)

    return env
