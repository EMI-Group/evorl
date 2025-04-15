import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from brax.envs import (
    Env as BraxEnv,
    get_environment,
)

from evorl.types import Action, PyTreeDict

from .env import Env, EnvAdapter, EnvState
from .space import Box, Space, SpaceContainer
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
    """Adapter for Brax environments."""

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
        obs_spec = self.env.observation_size

        def get_space(obs_size):
            if not isinstance(obs_size, tuple):
                obs_size = (obs_size,)
            obs_spec = jnp.full(obs_size, 1e10, dtype=jnp.float32)
            return Box(low=-obs_spec, high=obs_spec)

        if isinstance(obs_spec, int):
            return get_space(obs_spec)
        else:
            return SpaceContainer(
                spaces=jtu.tree_map(
                    get_space,
                    obs_spec,
                    is_leaf=lambda obj: isinstance(obj, tuple)
                    and all(isinstance(x, int) for x in obj),
                )
            )


def create_brax_env(env_name: str, **kwargs) -> BraxAdapter:
    """Create Brax environment.

    Args:
        env_name: Environment name.
        kwargs: Arguments passing into Brax.

    Returns:
        Brax env.
    """
    env = get_environment(env_name, **kwargs)
    env = BraxAdapter(env)

    return env


def create_wrapped_brax_env(
    env_name: str,
    episode_length: int = 1000,
    parallel: int = 1,
    autoreset_mode: AutoresetMode = AutoresetMode.NORMAL,
    discount: float | None = 1.0,
    record_ori_obs: bool = False,
    **kwargs,
) -> Env:
    """Create wrapped Brax environment for training.

    Args:
        env_name: Environment name.
        episode_length: Max episode length.
        parallel: Number of parallel environments.
        autoreset_mode: Autoreset mode.
        discount: Discount factor.
        record_ori_obs: Whether record original observation in AutoresetMode.NORMAL and AutoresetMode.FAST mode.
        kwargs: Other arguments passing into Brax.

    Returns:
        Wrapped Brax env.

    """
    env = create_brax_env(env_name, **kwargs)

    if autoreset_mode == AutoresetMode.ENVPOOL:
        # envpool mode will always record last obs
        record_ori_obs = False

    if autoreset_mode != AutoresetMode.DISABLED:
        env = EpisodeWrapper(
            env,
            episode_length,
            record_ori_obs=record_ori_obs,
            discount=discount,
        )
        if autoreset_mode == AutoresetMode.FAST:
            env = FastVmapAutoResetWrapper(env, num_envs=parallel)
        elif autoreset_mode == AutoresetMode.NORMAL:
            env = VmapAutoResetWrapper(env, num_envs=parallel)
        elif autoreset_mode == AutoresetMode.ENVPOOL:
            env = VmapEnvPoolAutoResetWrapper(env, num_envs=parallel)
    else:
        env = OneEpisodeWrapper(env, episode_length, record_ori_obs=record_ori_obs)
        env = VmapWrapper(env, num_envs=parallel, vmap_step=True)

    return env
