import warnings
import numpy as np
import math
import gymnasium
import multiprocessing as mp

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import Action, PyTreeDict

from .env import Env, EnvAdapter, EnvState
from .space import Box, Discrete, Space, SpaceContainer
from .utils import check_host_env_spaces, make_batched_io_callback
from .wrappers import Wrapper, AutoresetMode


def _to_jax(pytree):
    return jtu.tree_map(lambda x: jnp.asarray(x), pytree)


def _to_jax_spec(pytree):
    pytree = _to_jax(pytree)
    return jtu.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), pytree)


def _reshape_batch_dims(pytree, batch_shape):
    # [B1*...*Bn*#envs, *] -> [B1, ..., Bn, #envs, *]
    return jtu.tree_map(lambda x: np.reshape(x, batch_shape + x.shape[1:]), pytree)


class GymnasiumAdapter(EnvAdapter):
    """Adapter for Gymnasium to support Gymnasium environments.

    This env already is a vectorized environment and has experimental supports. It is not recommended to direcly replace other jax-based envs with this env in EvoRL's existing workflows. Users should carefully check the compatibility and modify the corresponding code to avoid the side-effects and other undefined behaviors.

    :::{caution}
    This env breaks the rule of pure functions. Its env state is maintained inside the gymnasium. Thesefore, users should use it with caution. Unlike other jax-based envs, this env has following limitations:

    - No support for recovering from previous env state.
        - In other word, you can't rewind after calling `env.step`.
        - For example, you can't resume the training from a checkpoint exactly as before; Similarly, `evorl.rollout.eval_rollout_episode` will also result in undefined behavior.
    - We use gymnasium's `AsyncVectorEnv`, which uses python's `multiprocessing` package for parallelism. This may cause performance issues, especially when the number of parallel environments is large.
        - We recommend that the total number of parallel environments does not exceed the number of CPU logic cores.
    :::
    """

    # TODO: multi-device support

    uses_host_callback = True

    def __init__(
        self,
        env_name: str,
        max_episode_steps: int,
        num_envs: int,
        record_ori_obs: bool = False,
        discount: float | None = None,
        vecenv_kwargs: dict | None = None,
        **env_kwargs,
    ):
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.num_envs = num_envs
        self.record_ori_obs = record_ori_obs
        self.record_episode_return = discount is not None
        self.discount = discount
        self.vecenv_kwargs = vecenv_kwargs
        self.env_specs = env_kwargs

        def _env_fn(num_envs):
            env = gymnasium.make_vec(
                self.env_name,
                num_envs=num_envs,
                max_episode_steps=self.max_episode_steps,
                vectorization_mode=gymnasium.VectorizeMode.ASYNC,
                vector_kwargs=self.vecenv_kwargs,
                **self.env_specs,
            )
            try:
                check_host_env_spaces(
                    env.single_observation_space, env.single_action_space
                )
                if env.metadata["autoreset_mode"] not in (
                    gymnasium.vector.AutoresetMode.NEXT_STEP,
                    gymnasium.vector.AutoresetMode.SAME_STEP,
                ):
                    raise NotImplementedError(
                        "Gymnasium adapter requires next-step or same-step autoreset."
                    )
            except Exception:
                env.close()
                raise
            return env

        self._env_fn = _env_fn
        self.env = _env_fn(num_envs)
        self.autoreset_mode = self.env.metadata["autoreset_mode"]

        self.setup_env_callback()

    def setup_env_callback(self):
        dummy_obs, _ = self.env.reset()
        # define your own dummy reset info here
        dummy_reset_info = PyTreeDict()
        if self.record_ori_obs:
            dummy_reset_info.ori_obs = dummy_obs
        # Preserve mapped state axes even when every member uses constant actions.
        dummy_token = np.zeros((self.num_envs,), dtype=np.int32)
        reset_spec = _to_jax_spec((dummy_obs, dummy_reset_info, dummy_token))

        dummy_action = self.env.single_action_space.sample()
        action_ndim = dummy_action.ndim
        dummy_actions = np.broadcast_to(
            dummy_action, (self.num_envs,) + dummy_action.shape
        )
        # define your own dummy step info here
        dummy_step_info = PyTreeDict()
        if self.record_ori_obs:
            dummy_step_info.ori_obs = dummy_obs

        step_spec = _to_jax_spec(
            self.env.step(dummy_actions)[:-1] + (dummy_step_info, dummy_token)
        )

        def _reset(key):
            batch_shape = key.shape[:-1]
            num_envs = math.prod(batch_shape) * self.num_envs

            self.env.close()
            self.env = self._env_fn(num_envs)

            assert self.env.num_envs == num_envs

            obs, _info = _reshape_batch_dims(
                self.env.reset(), batch_shape + (self.num_envs,)
            )
            # drop the original info dict as they do not have static shape.
            info = PyTreeDict()
            if self.record_ori_obs:
                info.ori_obs = jtu.tree_map(np.zeros_like, obs)

            token = np.zeros((*batch_shape, self.num_envs), dtype=np.int32)
            return obs, info, token

        def _step(token, actions):
            # [B1, ..., Bn, #envs]
            batch_ndim = actions.ndim - action_ndim
            batch_shape = actions.shape[:batch_ndim]

            # [B1, ..., Bn, #envs, *] -> [B1*...*Bn*#envs, *]
            actions = np.asarray(actions).reshape((-1, *actions.shape[batch_ndim:]))

            # [B1*...*Bn*#envs, *]
            obs, reward, termination, truncation, _info = self.env.step(actions)

            # drop the original info dict as they do not have static shape.
            info = PyTreeDict()
            if self.record_ori_obs:
                ori_obs = jtu.tree_map(np.copy, obs)
                final_obs_list = _info.get("final_obs", None)
                if final_obs_list is not None:
                    valid_indices = np.array(
                        [o is not None for o in final_obs_list]
                    ).nonzero()[0]
                    if valid_indices.size:
                        final_obs = jtu.tree_map(
                            lambda *values: np.stack(values),
                            *[final_obs_list[i] for i in valid_indices],
                        )
                        # Replace each leaf only for environments that ended.
                        for current, final in zip(
                            jtu.tree_leaves(ori_obs),
                            jtu.tree_leaves(final_obs),
                            strict=True,
                        ):
                            current[valid_indices] = final

                info.ori_obs = ori_obs

            return (
                *_reshape_batch_dims(
                    (obs, reward, termination, truncation, info), batch_shape
                ),
                np.asarray(token) + np.int32(1),
            )

        self._reset = make_batched_io_callback(_reset, reset_spec)
        self._step = make_batched_io_callback(_step, step_spec)

    def reset(self, key: jax.Array) -> EnvState:
        obs, info, token = _to_jax(self._reset(jax.random.key_data(key)))
        if isinstance(obs, dict):
            obs = PyTreeDict(obs)

        info.steps = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        info.termination = jnp.zeros((self.num_envs,))
        info.truncation = jnp.zeros((self.num_envs,))

        if self.autoreset_mode == gymnasium.vector.AutoresetMode.NEXT_STEP:
            info.autoreset = jnp.zeros((self.num_envs,))
        if self.record_episode_return:
            info.episode_return = jnp.zeros((self.num_envs,))

        return EnvState(
            env_state=None,
            obs=obs,
            reward=jnp.zeros((self.num_envs,)),
            done=jnp.zeros((self.num_envs,)),
            info=info,
            _internal=PyTreeDict(callback_token=token),
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        if self.autoreset_mode == gymnasium.vector.AutoresetMode.NEXT_STEP:
            return self._envpool_autoreset_step(state, action)
        elif self.autoreset_mode == gymnasium.vector.AutoresetMode.SAME_STEP:
            return self._normal_autoreset_step(state, action)
        else:
            raise NotImplementedError(
                f"Unsupported autoreset mode: {self.autoreset_mode}"
            )

    def _envpool_autoreset_step(self, state: EnvState, action: Action) -> EnvState:
        """Step for Next-Step mode."""
        autorest = state.done  # True = this step is the reset() step

        obs, reward, termination, truncation, info, token = _to_jax(
            self._step(state._internal.callback_token, action)
        )
        if isinstance(obs, dict):
            obs = PyTreeDict(obs)

        reward = reward.astype(jnp.float32)
        done = jnp.logical_or(termination, truncation).astype(jnp.float32)

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

        return state.replace(
            obs=obs,
            reward=reward,
            done=done,
            info=info,
            _internal=state._internal.replace(callback_token=token),
        )

    def _normal_autoreset_step(self, state: EnvState, action: Action) -> EnvState:
        """Step for Same-Step mode."""
        prev_done = state.done

        steps = state.info.steps * (1 - prev_done).astype(jnp.int32)
        if self.record_episode_return:
            episode_return = state.info.episode_return * (1 - prev_done)

        obs, reward, termination, truncation, info, token = _to_jax(
            self._step(state._internal.callback_token, action)
        )
        if isinstance(obs, dict):
            obs = PyTreeDict(obs)
        steps = steps + 1
        reward = reward.astype(jnp.float32)
        done = jnp.logical_or(termination, truncation).astype(jnp.float32)

        info.steps = steps
        info.termination = termination.astype(jnp.float32)
        info.truncation = truncation.astype(jnp.float32)
        if self.record_episode_return:
            if self.discount == 1.0:
                episode_return += reward
            else:
                episode_return += jnp.power(self.discount, steps - 1) * reward
            info.episode_return = episode_return

        return state.replace(
            obs=obs,
            reward=reward,
            done=done,
            info=info,
            _internal=state._internal.replace(callback_token=token),
        )

    @property
    def action_space(self) -> Space:
        return gymnasium_space_to_evorl_space(self.env.single_action_space)

    @property
    def obs_space(self) -> Space:
        return gymnasium_space_to_evorl_space(self.env.single_observation_space)


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


def gymnasium_space_to_evorl_space(space: gymnasium.Space) -> Space:
    if isinstance(space, gymnasium.spaces.Box):
        low = _inf_to_num(jnp.asarray(space.low))
        high = _inf_to_num(jnp.asarray(space.high))
        return Box(low=low, high=high)
    elif isinstance(space, gymnasium.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gymnasium.spaces.Dict):
        return SpaceContainer(
            spaces=PyTreeDict(
                {
                    key: gymnasium_space_to_evorl_space(child)
                    for key, child in space.spaces.items()
                }
            )
        )
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")


def create_gymnasium_env(
    env_name,
    episode_length: int = 1000,
    parallel: int = 1,
    autoreset_mode: AutoresetMode = AutoresetMode.ENVPOOL,
    discount: float | None = 1.0,
    record_ori_obs: bool = False,
    **kwargs,
) -> GymnasiumAdapter:
    """Create a gym env based on Gymnasium.

    Unlike other jax-based env, most wrappers are handled inside the gymnasium.
    """
    match autoreset_mode:
        case AutoresetMode.FAST:
            warnings.warn(
                f"{autoreset_mode} is not supported for Gymnasium Envs. Fallback to AutoresetMode.NORMAL.",
            )
            gymnasium_autoreset_mode = gymnasium.vector.AutoresetMode.SAME_STEP
        case AutoresetMode.NORMAL:
            gymnasium_autoreset_mode = gymnasium.vector.AutoresetMode.SAME_STEP
        case AutoresetMode.ENVPOOL:
            gymnasium_autoreset_mode = gymnasium.vector.AutoresetMode.NEXT_STEP
            if record_ori_obs:
                warnings.warn(
                    f"{autoreset_mode} does not need record_ori_obs. Fallback to False.",
                )
                record_ori_obs = False
        case AutoresetMode.DISABLED:
            gymnasium_autoreset_mode = gymnasium.vector.AutoresetMode.NEXT_STEP
            discount = None

    mp.get_start_method("spawn")
    vecenv_kwargs = dict(
        autoreset_mode=gymnasium_autoreset_mode,
        context="spawn",  # jax's os.fork() warning remains
    )
    if "vecenv_kwargs" in kwargs:
        vecenv_kwargs.update(kwargs.pop("vecenv_kwargs"))

    env = GymnasiumAdapter(
        env_name=env_name,
        max_episode_steps=episode_length,
        num_envs=parallel,
        record_ori_obs=record_ori_obs,
        discount=discount,
        vecenv_kwargs=vecenv_kwargs,
        **kwargs,
    )

    if autoreset_mode == AutoresetMode.DISABLED:
        env = OneEpisodeWrapper(env)

    return env


# Note: for env of Humanoid and HumanoidStandup, the action sapce is [-0.4, 0.4], we don't explicitly handle it. You need to manually squash the action space to [-1, 1] by using `ActionSquashWrapper`.
