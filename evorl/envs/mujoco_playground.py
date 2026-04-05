import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from mujoco_playground import registry, MjxEnv

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


def _normalize_obs(obs):
    """Normalize obs to PyTreeDict if it's a raw dict.

    Some envs (e.g., mujoco_playground) return dict obs.
    PyTreeDict.__setattr__ auto-converts nested dicts, so we normalize
    upfront to avoid pytree node type mismatches in downstream wrappers.
    """
    if isinstance(obs, dict) and not isinstance(obs, PyTreeDict):
        return PyTreeDict(obs)
    return obs


class MjxEnvAdapter(EnvAdapter):
    """Adapter for Mujoco Playground environments."""

    def __init__(self, env: MjxEnv):
        super().__init__(env)
        # Cache _impl from a dummy reset. We strip _impl from the state to
        # save memory (intermediate computation results) and to avoid
        # non-vmappable fields (warp backend) breaking jax.lax.map.
        # The cached _impl is restored before each env.step() call.
        dummy_state = self.env.reset(jax.random.PRNGKey(0))
        self._static_impl = getattr(dummy_state.data, "_impl", None)

    def reset(self, key: chex.PRNGKey) -> EnvState:
        key, reset_key = jax.random.split(key)
        mjxenv_state = self.env.reset(reset_key)

        # Strip _impl for vmap compatibility with jax.lax.map
        if self._static_impl is not None:
            mjxenv_state = mjxenv_state.replace(data=mjxenv_state.data.replace(_impl=None))

        info = PyTreeDict(sort_dict(mjxenv_state.info))
        info.metrics = PyTreeDict(sort_dict(mjxenv_state.metrics))

        return EnvState(
            env_state=mjxenv_state,
            obs=_normalize_obs(mjxenv_state.obs),
            reward=mjxenv_state.reward,
            done=mjxenv_state.done,
            info=info,
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        mjxenv_state = state.env_state
        if self._static_impl is not None:
            mjxenv_state = mjxenv_state.replace(data=mjxenv_state.data.replace(_impl=self._static_impl))

        mjxenv_state = self.env.step(mjxenv_state, action)

        if self._static_impl is not None:
            mjxenv_state = mjxenv_state.replace(data=mjxenv_state.data.replace(_impl=None))

        metrics = state.info.metrics.replace(**mjxenv_state.metrics)

        info = state.info.replace(**mjxenv_state.info, metrics=metrics)

        return state.replace(
            env_state=mjxenv_state,
            obs=_normalize_obs(mjxenv_state.obs),
            reward=mjxenv_state.reward,
            done=mjxenv_state.done,
            info=info,
        )

    @property
    def action_space(self) -> Space:
        action_spec = jnp.tile(jnp.array([-1.0, 1.0]), (self.env.action_size, 1))
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


def create_mujoco_playground_env(env_name: str, **kwargs) -> MjxEnvAdapter:
    """Create Brax environment.

    Args:
        env_name: Environment name.
        kwargs: Arguments passing into Brax.

    Returns:
        Brax env.
    """
    # if "config_overrides" not in kwargs:
    #     kwargs["config_overrides"] = {}
    
    # if "impl" not in kwargs["config_overrides"]:
    #     kwargs["config_overrides"]["impl"] = "jax"

    env = registry.load(env_name, **kwargs)
    env = MjxEnvAdapter(env)

    return env


def create_wrapped_mujoco_playground_env(
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
    env = create_mujoco_playground_env(env_name, **kwargs)

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
