import chex
import gymnax
import jax
import jax.numpy as jnp
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.spaces import Box as GymnaxBox
from gymnax.environments.spaces import Discrete as GymnaxDiscrete
from gymnax.environments.spaces import Space as GymnaxSpace

from evorl.types import Action, PyTreeDict

from .env import Env, EnvAdapter, EnvState
from .space import Box, Discrete, Space
from .wrappers.action_wrapper import ActionSquashWrapper
from .wrappers.obs_wrapper import ObsFlattenWrapper
from .wrappers.training_wrapper import (
    AutoresetMode,
    EpisodeWrapper,
    FastVmapAutoResetWrapper,
    OneEpisodeWrapper,
    VmapAutoResetWrapper,
    VmapEnvPoolAutoResetWrapper,
    VmapWrapper,
)


class GymnaxAdapter(EnvAdapter):
    """Adapter for Gymnax environments."""

    def __init__(self, env: GymnaxEnv, env_params: chex.ArrayTree | None = None):
        super().__init__(env)
        self.env_params = env_params or env.default_params

    def reset(self, key: chex.PRNGKey) -> EnvState:
        key, reset_key = jax.random.split(key)
        obs, env_state = self.env.reset(reset_key, self.env_params)

        info = PyTreeDict(
            discount=jnp.ones(()),
            env_params=self.env_params,
        )

        return EnvState(
            env_state=env_state,
            obs=obs,
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            info=info,
            _internal=PyTreeDict(step_key=key),
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        key, step_key = jax.random.split(state._internal.step_key)

        # call step_env() instead of step() to disable autoreset
        # we handle the autoreset at AutoResetWrapper
        obs, env_state, reward, done, info = self.env.step_env(
            step_key, state.env_state, action, state.info.env_params
        )
        reward = reward.astype(jnp.float32)
        done = done.astype(jnp.float32)

        info = state.info.replace(
            discount=info["discount"].astype(jnp.float32),
        )
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
    def action_space(self) -> Space:
        return gymnax_space_to_evorl_space(self.env.action_space(self.env_params))

    @property
    def obs_space(self) -> Space:
        return gymnax_space_to_evorl_space(self.env.observation_space(self.env_params))


def _inf_to_num(x, num=1e10):
    return jnp.nan_to_num(x, posinf=num, neginf=-num)


def gymnax_space_to_evorl_space(space: GymnaxSpace):
    if isinstance(space, GymnaxBox):
        low = _inf_to_num(jnp.broadcast_to(space.low, space.shape).astype(space.dtype))
        high = _inf_to_num(
            jnp.broadcast_to(space.high, space.shape).astype(space.dtype)
        )
        return Box(low=low, high=high)
    elif isinstance(space, GymnaxDiscrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")


# image_env_list = [
#     "Asterix-MinAtar",
#     "Breakout-MinAtar",
#     "Freeway-MinAtar",
#     # "Seaquest-MinAtar",
#     "SpaceInvaders-MinAtar",
#     "MNISTBandit-bsuite",
#     "Pong-misc",
# ]


def create_gymnax_env(
    env_name: str, flatten_obs: bool = True, **kwargs
) -> GymnaxAdapter:
    env, env_params = gymnax.make(env_name)

    update_env_params = {k: v for k, v in kwargs.items() if hasattr(env_params, k)}
    env_params = env_params.replace(**update_env_params)

    env = GymnaxAdapter(env, env_params)

    if isinstance(env.action_space, Box):
        if not jnp.logical_and(
            (env.action_space.low == -1).all(), (env.action_space.high == 1).all()
        ):
            env = ActionSquashWrapper(env)

    if flatten_obs and len(env.obs_space.shape) > 1:
        env = ObsFlattenWrapper(env)

    return env


def create_wrapped_gymnax_env(
    env_name: str,
    flatten_obs: bool = True,
    episode_length: int = 1000,
    parallel: int = 1,
    autoreset_mode: AutoresetMode = AutoresetMode.NORMAL,
    discount: float | None = 1.0,
    record_ori_obs: bool = False,
    **kwargs,
) -> Env:
    env = create_gymnax_env(env_name, flatten_obs, **kwargs)

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
