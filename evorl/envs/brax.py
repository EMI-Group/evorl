import chex
import jax
import jax.numpy as jnp
from brax.envs import Env as BraxEnv
from brax.envs import get_environment

from evorl.types import Action, PyTreeDict

from .env import Env, EnvAdapter, EnvState
from .space import Box, Space
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


class BraxAdapter(EnvAdapter):
    def __init__(self, env: BraxEnv):
        super().__init__(env)

    def reset(self, key: chex.PRNGKey) -> EnvState:
        key, reset_key = jax.random.split(key)
        brax_state = self.env.reset(reset_key)

        info = PyTreeDict(sort_dict(brax_state.info))
        info.metrics = PyTreeDict(sort_dict(brax_state.metrics))

        return EnvState(
            env_state=brax_state,
            obs=brax_state.obs,
            reward=brax_state.reward,
            done=brax_state.done,
            info=info,
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        brax_state = self.env.step(state.env_state, action)

        metrics = state.info.metrics.replace(**brax_state.metrics)

        info = state.info.replace(**brax_state.info, metrics=metrics)

        return state.replace(
            env_state=brax_state,
            obs=brax_state.obs,
            reward=brax_state.reward,
            done=brax_state.done,
            info=info,
        )

    @property
    def action_space(self) -> Space:
        action_spec = jnp.asarray(self.env.sys.actuator.ctrl_range, dtype=jnp.float32)
        return Box(low=action_spec[:, 0], high=action_spec[:, 1])

    @property
    def obs_space(self) -> Space:
        obs_spec = jnp.full((self.env.observation_size,), 1e10, dtype=jnp.float32)
        return Box(low=-obs_spec, high=obs_spec)


def create_brax_env(env_name: str, **kwargs) -> BraxAdapter:
    """
    Args:
        Autoreset: When use envs for RL training, set autoreset=True. When use envs for evaluation, set autoreset=False.
        discount: discount factor for episode return calculation. The episode returns are Only recorded when autoreset=True.
    """

    env = get_environment(env_name, **kwargs)
    env = BraxAdapter(env)

    return env


def create_wrapped_brax_env(
    env_name: str,
    episode_length: int = 1000,
    parallel: int = 1,
    autoreset_mode: AutoresetMode = AutoresetMode.NORMAL,
    discount: float = 1.0,
    record_last_obs: bool = False,
    **kwargs,
) -> Env:
    env = create_brax_env(env_name, **kwargs)

    if autoreset_mode == AutoresetMode.ENVPOOL:
        # envpool mode will always record last obs
        record_last_obs = False

    if autoreset_mode != AutoresetMode.DISABLED:
        env = EpisodeWrapper(
            env,
            episode_length,
            record_last_obs=record_last_obs,
            record_episode_return=True,
            discount=discount,
        )
        if autoreset_mode == AutoresetMode.FAST:
            env = FastVmapAutoResetWrapper(env, num_envs=parallel)
        elif autoreset_mode == AutoresetMode.NORMAL:
            env = VmapAutoResetWrapper(env, num_envs=parallel)
        elif autoreset_mode == AutoresetMode.ENVPOOL:
            env = VmapEnvPoolAutoResetWrapper(env, num_envs=parallel)
    else:
        env = OneEpisodeWrapper(env, episode_length, record_last_obs=record_last_obs)
        env = VmapWrapper(env, num_envs=parallel, vmap_step=True)

    return env
