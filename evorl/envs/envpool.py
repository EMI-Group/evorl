from functools import partial
import warnings

import chex
import envpool
import gym
import gym.spaces
import gymnasium
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import math

from evorl.types import Action, PyTreeDict

from .env import Env, EnvAdapter, EnvState
from .space import Box, Discrete, Space
from .wrappers import Wrapper, AutoresetMode


def _to_jax(pytree):
    return jtu.tree_map(lambda x: jnp.asarray(x), pytree)


def _to_jax_spec(pytree):
    pytree = _to_jax(pytree)
    return jtu.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), pytree)


def _reshape_batch_dims(pytree, batch_shape):
    # [B1*...*Bn*#envs, *] -> [B1, ..., Bn, #envs, *]
    return jtu.tree_map(lambda x: jnp.reshape(x, batch_shape + x.shape[1:]), pytree)


class EnvPoolGymAdapter(EnvAdapter):
    """Adapter for EnvPool to support EnvPool environments.

    This env already is a vectorized environment and has experimental supports. It is not recommended to direcly replace other jax-based envs with this env in EvoRL's existing workflows. Users should carefully check the compatibility and modify the corresponding code to avoid the side-effects and other undefined behaviors.

    :::{caution}
    This env breaks the rule of pure functions. Its env state is maintained inside the envpool. Thesefore, users should use it with caution. Unlike other jax-based envs, this env has following limitations:

    - No support for recovering from previous env state.
        - In other word, you can't rewind after calling `env.step`.
        - For example, you can't resume the training from a checkpoint exactly as before; Similarly, `evorl.rollout.eval_rollout_episode` will also result in undefined behavior.
    :::
    """

    # TODO: multi-device support

    def __init__(
        self,
        env_name: str,
        env_backend: str,
        max_episode_steps: int,
        num_envs: int,
        discount: float | None = None,
        **env_kwargs,
    ):
        self.env_name = env_name
        self.env_backend = env_backend
        self.max_episode_steps = max_episode_steps
        self.num_envs = num_envs
        self.record_episode_return = discount is not None
        self.discount = discount
        self.env_kwargs = env_kwargs

        def _env_fn(_num_envs):
            if self.env_backend == "gymnasium":
                env = envpool.make_gymnasium(
                    self.env_name,
                    num_envs=_num_envs,
                    max_episode_steps=max_episode_steps,
                    **env_kwargs,
                )
            elif self.env_backend == "gym":
                env = envpool.make_gym(
                    self.env_name,
                    num_envs=_num_envs,
                    max_episode_steps=max_episode_steps,
                    **env_kwargs,
                )
            else:
                raise ValueError(f"Unsupported env_backend: {self.env_backend}")

            return env

        self._env_fn = _env_fn
        self.env = _env_fn(self.num_envs)

        self.setup_env_callback()

    def setup_env_callback(self):
        dummy_obs, _ = self.env.reset()
        # define your own dummy reset info
        dummy_reset_info = PyTreeDict()
        reset_spec = _to_jax_spec((dummy_obs, dummy_reset_info))

        dummy_action = self.env.action_space.sample()
        dummy_actions = np.broadcast_to(
            dummy_action, (self.num_envs,) + dummy_action.shape
        )
        # define your own dummy step info
        dummy_step_info = PyTreeDict()
        step_spec = _to_jax_spec(self.env.step(dummy_actions)[:-1] + (dummy_step_info,))

        def _reset(key):
            batch_shape = key.shape[:-1]
            num_envs = math.prod(batch_shape) * self.num_envs

            self.env = self._env_fn(num_envs)

            assert self.env.config["num_envs"] == num_envs

            obs, _info = _reshape_batch_dims(
                self.env.reset(), batch_shape + (self.num_envs,)
            )

            # drop the original info dict as they do not have static shape.
            info = PyTreeDict()

            return obs, info

        def _step(actions):
            # Note: we are not sure if self.env is always updated by _reset in JIT mode.

            # [B1, ..., Bn, #envs]
            batch_shape = actions.shape[: -len(self.action_space.shape)]

            # [B1, ..., Bn, #envs, *] -> [B1*...*Bn*#envs, *]
            actions = jax.lax.collapse(actions, 0, len(batch_shape))

            obs, reward, termination, truncation, _info = self.env.step(
                np.asarray(actions)
            )

            # drop the original info dict as they do not have static shape.
            info = PyTreeDict()

            return _reshape_batch_dims(
                (obs, reward, termination, truncation, info), batch_shape
            )

        # You are entring the dangerous zone!!!
        # _reset and _step are not pure functions. Use with caution.
        self._reset = partial(
            jax.pure_callback, _reset, reset_spec, vmap_method="expand_dims"
        )
        self._step = partial(
            jax.pure_callback, _step, step_spec, vmap_method="expand_dims"
        )

    def reset(self, key: chex.PRNGKey) -> EnvState:
        obs, info = _to_jax(self._reset(key))

        info.steps = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        info.termination = jnp.zeros((self.num_envs,))
        info.truncation = jnp.zeros((self.num_envs,))
        info.episode_return = jnp.zeros((self.num_envs,))
        info.autoreset = jnp.zeros((self.num_envs,))

        if self.record_episode_return:
            info.episode_return = jnp.zeros((self.num_envs,))

        return EnvState(
            env_state=None,
            obs=obs,
            reward=jnp.zeros((self.num_envs,)),
            done=jnp.zeros((self.num_envs,)),
            info=info,
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        autorest = state.done  # True = this step is the reset() step

        obs, reward, termination, truncation, info = _to_jax(self._step(action))

        reward = reward.astype(jnp.float32)
        done = (jnp.logical_or(termination, truncation)).astype(jnp.float32)

        info.steps = (state.info.steps + 1) * (1 - autorest).astype(jnp.int32)
        info.termination = termination.astype(jnp.float32)
        info.truncation = truncation.astype(jnp.float32)
        info.autoreset = autorest  # prev_done

        if self.record_episode_return:
            episode_return = state.info.episode_return
            if self.discount == 1.0:
                episode_return += reward
            else:
                episode_return += jnp.power(self.discount, state.info.steps) * reward
            info.episode_return = episode_return * (1 - autorest)

        return state.replace(obs=obs, reward=reward, done=done, info=info)

    @property
    def action_space(self) -> Space:
        return gym_space_to_evorl_space(self.env.action_space)

    @property
    def obs_space(self) -> Space:
        return gym_space_to_evorl_space(self.env.observation_space)


# TODO: EnvPoolDMAdapter
class OneEpisodeWrapper(Wrapper):
    """Vectorized one-episode wrapper for evaluation."""

    def __init__(self, env: Env):
        super().__init__(env)

    def step(self, state: EnvState, action: Action) -> EnvState:
        # Note: could add extra CPU overhead

        def where_done(x, y):
            done = state.done
            if done.ndim > 0:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        return jtu.tree_map(
            where_done,
            state,
            self.env.step(state, action),
        )


def _inf_to_num(x, num=1e10):
    return jnp.nan_to_num(x, posinf=num, neginf=-num)


def gym_space_to_evorl_space(space: gymnasium.Space | gym.Space) -> Space:
    if isinstance(space, gymnasium.spaces.Box) or isinstance(space, gym.spaces.Box):
        low = _inf_to_num(jnp.asarray(space.low))
        high = _inf_to_num(jnp.asarray(space.high))
        return Box(low=low, high=high)
    elif isinstance(space, gymnasium.spaces.Discrete) or isinstance(
        space, gym.spaces.Discrete
    ):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")


def create_envpool_env(
    env_name,
    env_backend: str = "gymnasium",
    episode_length: int = 1000,
    parallel: int = 1,
    autoreset_mode: AutoresetMode = AutoresetMode.ENVPOOL,
    discount: float | None = 1.0,
    **kwargs,
) -> EnvPoolGymAdapter:
    """Create a gym env based on EnvPool.

    Unlike other jax-based env, most wrappers are handled inside the envpool.
    """
    match autoreset_mode:
        case AutoresetMode.NORMAL | AutoresetMode.FAST:
            warnings.warn(
                f"{autoreset_mode} is not supported for EnvPool Envs. Fallback to AutoresetMode.ENVPOOL.",
            )
            autoreset_mode = AutoresetMode.ENVPOOL
        case AutoresetMode.DISABLED:
            discount = None

    if env_backend in ["gym", "gymnasium"]:
        env = EnvPoolGymAdapter(
            env_name=env_name,
            env_backend=env_backend,
            max_episode_steps=episode_length,
            num_envs=parallel,
            discount=discount,
            **kwargs,
        )
    else:
        raise ValueError(f"env_backend {env_backend} is not supported")

    if autoreset_mode == AutoresetMode.DISABLED:
        env = OneEpisodeWrapper(env)

    return env


# Note: for env of Humanoid and HumanoidStandup, the action sapce is [-0.4, 0.4], we don't explicitly handle it. You need to manually squash the action space to [-1, 1] by using `ActionSquashWrapper`.
