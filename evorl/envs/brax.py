import jax
import jax.numpy as jnp

from flax import struct
import chex
from .env import EnvAdapter, EnvState, Env
from .space import Space, Box
from .utils import sort_dict
from evorl.types import Action, PyTreeDict
from brax.envs import (
    Env as BraxEnv,
    get_environment
)

from .wrappers.training_wrapper import EpisodeWrapper, OneEpisodeWrapper, VmapAutoResetWrapper, VmapWrapper, FastVmapAutoResetWrapper


class BraxAdapter(EnvAdapter):
    def __init__(self, env: BraxEnv):
        super(BraxAdapter, self).__init__(env)

        action_spec = jnp.asarray(
            env.sys.actuator.ctrl_range, dtype=jnp.float32)
        self._action_sapce = Box(low=action_spec[:, 0], high=action_spec[:, 1])

        # Note: use jnp.inf or jnp.finfo(jnp.float32).min|max causes inf
        obs_spec = jnp.full((env.observation_size,),
                            1e10, dtype=jnp.float32)
        self._obs_space = Box(low=-obs_spec, high=obs_spec)

    def reset(self, key: chex.PRNGKey) -> EnvState:
        key, reset_key = jax.random.split(key)
        brax_state = self.env.reset(reset_key)

        info = PyTreeDict(sort_dict(brax_state.info))
        info.metrics = PyTreeDict(sort_dict(brax_state.metrics))
        # not necessary, but we need non-empty extra until orbax fixes #818
        extra = PyTreeDict(step_key=key) 

        return EnvState(
            env_state=brax_state,
            obs=brax_state.obs,
            reward=brax_state.reward,
            done=brax_state.done,
            info=info,
            extra=extra
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        brax_state = self.env.step(state.env_state, action)

        state.info.update(brax_state.info)
        state.info.metrics.update(brax_state.metrics)

        return state.replace(
            env_state=brax_state,
            obs=brax_state.obs,
            reward=brax_state.reward,
            done=brax_state.done,
        )

    @property
    def action_space(self) -> Space:
        return self._action_sapce

    @property
    def obs_space(self) -> Space:
        return self._obs_space


def create_brax_env(env_name: str, **kwargs) -> BraxAdapter:
    """
        Args:
            Autoreset: When use envs for RL training, set autoreset=True. When use envs for evaluation, set autoreset=False.
            discount: discount factor for episode return calculation. The episode returns are Only recorded when autoreset=True.
    """

    env = get_environment(env_name, **kwargs)
    env = BraxAdapter(env)

    return env


def create_wrapped_brax_env(env_name: str,
                            episode_length: int = 1000,
                            parallel: int = 1,
                            autoreset: bool = True,
                            fast_reset: bool = True,
                            discount: float = 1.0,
                            **kwargs) -> Env:
    env = create_brax_env(env_name, **kwargs)
    if autoreset:
        env = EpisodeWrapper(env, episode_length,
                             record_episode_return=True, discount=discount)
        if fast_reset:
            env = FastVmapAutoResetWrapper(env, num_envs=parallel)
        else:
            env = VmapAutoResetWrapper(env, num_envs=parallel)
    else:
        env = OneEpisodeWrapper(env, episode_length)
        env = VmapWrapper(env, num_envs=parallel, vmap_step=True)

    return env
