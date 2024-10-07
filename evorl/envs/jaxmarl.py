from collections.abc import Mapping

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments import MultiAgentEnv

from evorl.types import Action, AgentID, PyTreeDict
from evorl.utils.jax_utils import tree_astype, tree_zeros_like

from .env import EnvState
from .gymnax import gymnax_space_to_evorl_space
from .jaxmarl_envs import make_mabrax_env
from .multi_agent_env import MultiAgentEnvAdapter
from .space import Space
from .utils import sort_dict
from .wrappers.training_wrapper import (
    AutoresetMode,
    EpisodeWrapper,
    FastVmapAutoResetWrapper,
    OneEpisodeWrapper,
    VmapAutoResetWrapper,
    VmapEnvPoolAutoResetWrapper,
    VmapWrapper,
)


def get_random_actions(env: MultiAgentEnv):
    dummy_key = jax.random.PRNGKey(42)

    return {
        agent_id: env.action_space(agent_id).sample(dummy_key)
        for agent_id in env.agents
    }


class JaxMARLAdapter(MultiAgentEnvAdapter):
    def __init__(self, env: MultiAgentEnv):
        super().__init__(env)

    def reset(self, key: chex.PRNGKey) -> EnvState:
        key, reset_key, dummy_step_key = jax.random.split(key, 3)
        obs, env_state = self.env.reset(reset_key)

        dummy_action = get_random_actions(self.env)

        # run one dummy step to get reward,done,info shape
        _, _, dummy_reward, dummy_done, dummy_info = self.env.step_env(
            dummy_step_key, env_state, dummy_action
        )

        info = PyTreeDict(sort_dict(dummy_info))
        extra = PyTreeDict(step_key=key)

        return EnvState(
            env_state=env_state,
            obs=obs,
            reward=tree_zeros_like(dummy_reward, dtype=jnp.float32),
            done=tree_zeros_like(dummy_done, dtype=jnp.float32),
            info=info,
            _internal=extra,
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        key, step_key = jax.random.split(state._internal.step_key)

        # call step_env() instead of step() to disable autoreset
        # we handle the autoreset at AutoResetWrapper
        obs, env_state, reward, done, info = self.env.step_env(
            step_key, state.env_state, action
        )
        reward = tree_astype(reward, jnp.float32)
        done = tree_astype(done, jnp.float32)

        info = state.info.replace(**info)
        _internal = state._internal.replace(step_key=key)

        return state.replace(
            env_state=env_state,
            obs=obs,
            reward=reward,
            done=done,
            info=info,
            _internal=_internal,
        )

    @property
    def action_space(self) -> Mapping[AgentID, Space]:
        return jaxmarl_space_to_evorl_space(self.env.action_spaces)

    @property
    def obs_space(self) -> Mapping[AgentID, Space]:
        return jaxmarl_space_to_evorl_space(self.env.observation_spaces)

    @property
    def agents(self) -> list[AgentID]:
        return self.env.agents


def jaxmarl_space_to_evorl_space(space):
    return {
        agent_id: gymnax_space_to_evorl_space(space)
        for agent_id, space in space.items()
    }


supported_jaxmarl_env_list = (
    "MPE_simple_v3",
    "MPE_simple_tag_v3",
    "MPE_simple_world_comm_v3",
    "MPE_simple_spread_v3",
    "MPE_simple_crypto_v3",
    "MPE_simple_speaker_listener_v4",
    "MPE_simple_push_v3",
    "MPE_simple_adversary_v3",
    "MPE_simple_reference_v3",
    "MPE_simple_facmac_v1",
    "MPE_simple_facmac_3a_v1",
    "MPE_simple_facmac_6a_v1",
    "MPE_simple_facmac_9a_v1",
    # "switch_riddle",
    # "SMAX",
    # "HeuristicEnemySMAX",
    # "LearnedPolicyEnemySMAX",
    "ant_4x2",
    "halfcheetah_6x1",
    "hopper_3x1",
    "humanoid_9|8",
    "walker2d_2x3",
    # "storm",
    # "storm_2p",
    # "hanabi",
    # "overcooked",
    # "coin_game",
)

ma_brax_env_list = (
    "ant_4x2",
    "halfcheetah_6x1",
    "hopper_3x1",
    "humanoid_9|8",
    "walker2d_2x3",
)


class MABraxAdapter(JaxMARLAdapter):
    def reset(self, key: chex.PRNGKey) -> EnvState:
        state = super().reset(key)
        # setup global info
        state.info.global_obs = state.env_state.obs

        return state

    def step(self, state: EnvState, action: Action) -> EnvState:
        if not isinstance(action, dict):
            action = self.unwrapped.map_agents_to_global_action(action)

        state = super().step(state, action)
        # update global info
        state.info.global_obs = state.env_state.obs
        return state


def create_mabrax_env(env_name: str, **kwargs) -> JaxMARLAdapter:
    if env_name not in ma_brax_env_list:
        raise ValueError(f"Unsupported jaxmarl env: {env_name}")

    env = make_mabrax_env(env_name, **kwargs)

    # Note: mabrax internally use brax's traning wrapper (EpisodeWrapper)
    env = MABraxAdapter(env)

    return env


def create_wrapped_mabrax_env(
    env_name: str,
    episode_length: int = 1000,
    parallel: int = 1,
    autoreset_mode: AutoresetMode = AutoresetMode.NORMAL,
    **kwargs,
) -> JaxMARLAdapter:
    env = create_mabrax_env(env_name, **kwargs)
    if autoreset_mode != AutoresetMode.DISABLED:
        env = EpisodeWrapper(env, episode_length)
        if autoreset_mode == AutoresetMode.FAST:
            env = FastVmapAutoResetWrapper(env, num_envs=parallel)
        elif autoreset_mode == AutoresetMode.NORMAL:
            env = VmapAutoResetWrapper(env, num_envs=parallel)
        elif autoreset_mode == AutoresetMode.ENVPOOL:
            env = VmapEnvPoolAutoResetWrapper(env, num_envs=parallel)
    else:
        env = OneEpisodeWrapper(env, episode_length)
        env = VmapWrapper(env, num_envs=parallel, vmap_step=True)

    return env
