import envpool
import gymnasium
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import math

from evorl.types import Action, PyTreeDict

from .env import BaseEnv, EnvAdapter, EnvState
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


class EnvPoolGymAdapter(EnvAdapter):
    """Adapter for EnvPool to support EnvPool environments.

    `autoreset_next_step=True` uses EnvPool's native next-step autoreset.
    When `False`, ended rows are reset in the same host callback as the step.

    This env already is a vectorized environment and has experimental supports. It is not recommended to direcly replace other jax-based envs with this env in EvoRL's existing workflows. Users should carefully check the compatibility and modify the corresponding code to avoid the side-effects and other undefined behaviors.

    :::{caution}
    This env breaks the rule of pure functions. Its env state is maintained inside the envpool. Thesefore, users should use it with caution. Unlike other jax-based envs, this env has following limitations:

    - No support for recovering from previous env state.
        - In other word, you can't rewind after calling `env.step`.
        - For example, you can't resume the training from a checkpoint exactly as before; Similarly, `evorl.rollout.eval_rollout_episode` will also result in undefined behavior.
    :::
    """

    # TODO: multi-device support

    uses_host_callback = True

    def __init__(
        self,
        env_name: str,
        max_episode_steps: int,
        num_envs: int,
        discount: float | None = None,
        record_ori_obs: bool = False,
        autoreset_next_step: bool = True,
        **env_kwargs,
    ):
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.num_envs = num_envs
        self.record_episode_return = discount is not None
        self.discount = discount
        self.record_ori_obs = record_ori_obs
        self.autoreset_next_step = autoreset_next_step
        self.env_kwargs = env_kwargs

        def _env_fn(_num_envs):
            env = envpool.make_gymnasium(
                self.env_name,
                num_envs=_num_envs,
                max_episode_steps=max_episode_steps,
                **env_kwargs,
            )
            try:
                check_host_env_spaces(env.observation_space, env.action_space)
                if env.config["max_num_players"] != 1:
                    raise NotImplementedError(
                        "EnvPool adapter requires single-player tasks."
                    )
                if env.config["batch_size"] != _num_envs:
                    raise ValueError(
                        "EnvPool batch_size must equal the full environment count."
                    )
                if (
                    env.metadata.get("autoreset_mode")
                    == gymnasium.vector.AutoresetMode.SAME_STEP
                ):
                    raise NotImplementedError(
                        "EnvPool adapter requires next-step autoreset."
                    )
            except Exception:
                env.close()
                raise
            return env

        self._env_fn = _env_fn
        self.env = _env_fn(self.num_envs)

        self.setup_env_callback()

    def setup_env_callback(self):
        dummy_obs, _ = self.env.reset()
        # define your own dummy reset info
        dummy_reset_info = PyTreeDict()
        if self.record_ori_obs:
            dummy_reset_info.ori_obs = dummy_obs
        # Preserve mapped state axes even when every member uses constant actions.
        dummy_token = np.zeros((self.num_envs,), dtype=np.int32)
        reset_spec = _to_jax_spec((dummy_obs, dummy_reset_info, dummy_token))

        dummy_action = self.env.action_space.sample()
        action_ndim = dummy_action.ndim
        dummy_actions = np.broadcast_to(
            dummy_action, (self.num_envs,) + dummy_action.shape
        )
        # define your own dummy step info
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

            assert self.env.config["num_envs"] == num_envs

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

            obs, reward, termination, truncation, _info = self.env.step(actions)

            # drop the original info dict as they do not have static shape.
            info = PyTreeDict()
            if self.record_ori_obs:
                info.ori_obs = jtu.tree_map(np.copy, obs)

            if not self.autoreset_next_step:
                ended = np.flatnonzero(termination | truncation).astype(np.int32)
                if ended.size:
                    # Snapshot the transition before another native call can
                    # reuse its buffers. Reset only the environments that ended.
                    obs, reward, termination, truncation = jtu.tree_map(
                        np.copy, (obs, reward, termination, truncation)
                    )
                    reset_obs, reset_info = self.env.reset(ended)
                    reset_ids = np.asarray(reset_info["env_id"], dtype=np.int32)
                    for current, initial in zip(
                        jtu.tree_leaves(obs), jtu.tree_leaves(reset_obs), strict=True
                    ):
                        current[reset_ids] = initial

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
        if self.record_episode_return:
            info.episode_return = jnp.zeros((self.num_envs,))
        if self.autoreset_next_step:
            info.autoreset = jnp.zeros((self.num_envs,))

        return EnvState(
            env_state=None,
            obs=obs,
            reward=jnp.zeros((self.num_envs,)),
            done=jnp.zeros((self.num_envs,)),
            info=info,
            _internal=PyTreeDict(callback_token=token),
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        prev_done = state.done

        obs, reward, termination, truncation, info, token = _to_jax(
            self._step(state._internal.callback_token, action)
        )
        if isinstance(obs, dict):
            obs = PyTreeDict(obs)

        reward = reward.astype(jnp.float32)
        done = (jnp.logical_or(termination, truncation)).astype(jnp.float32)

        info.termination = termination.astype(jnp.float32)
        info.truncation = truncation.astype(jnp.float32)
        if not self.autoreset_next_step:
            # The previous step already reset ended environments; this is a
            # real transition, including the first step of each new episode.
            reward_steps = state.info.steps * (1 - prev_done).astype(jnp.int32)
            info.steps = reward_steps + 1
        else:
            reward_steps = state.info.steps
            info.steps = (state.info.steps + 1) * (1 - prev_done).astype(jnp.int32)
            info.autoreset = prev_done

        if self.record_episode_return:
            episode_return = state.info.episode_return
            if not self.autoreset_next_step:
                episode_return = episode_return * (1 - prev_done)
            if self.discount == 1.0:
                episode_return += reward
            else:
                episode_return += jnp.power(self.discount, reward_steps) * reward
            if self.autoreset_next_step:
                episode_return = episode_return * (1 - prev_done)
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
        return gym_space_to_evorl_space(self.env.action_space)

    @property
    def obs_space(self) -> Space:
        return gym_space_to_evorl_space(self.env.observation_space)


# TODO: EnvPoolDMAdapter
class OneEpisodeWrapper(Wrapper[Action, Space]):
    """Vectorized one-episode wrapper for evaluation."""

    def __init__(self, env: BaseEnv[Action, Space]):
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


def gym_space_to_evorl_space(space: gymnasium.Space) -> Space:
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
                    key: gym_space_to_evorl_space(child)
                    for key, child in space.spaces.items()
                }
            )
        )
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")


def create_envpool_env(
    env_name,
    episode_length: int = 1000,
    parallel: int = 1,
    autoreset_mode: AutoresetMode = AutoresetMode.ENVPOOL,
    discount: float | None = 1.0,
    record_ori_obs: bool = False,
    **kwargs,
) -> BaseEnv[Action, Space]:
    """Create an EnvPool environment with Gymnasium API.

    Unlike other jax-based env, most wrappers are handled inside the envpool.
    `NORMAL` resets ended sub-environments within the same ordered host callback,
    preserving their terminal reward and flags while returning reset observations.
    `ENVPOOL` uses native next-step autoreset. `FAST` is unsupported and raises
    `NotImplementedError`.
    `record_ori_obs=True` exposes the observation before reset in `info.ori_obs`.
    With next-step autoreset this is the observation returned by the step itself.
    """
    match autoreset_mode:
        case AutoresetMode.FAST:
            raise NotImplementedError(
                "EnvPool does not support AutoresetMode.FAST. "
                "Use NORMAL, ENVPOOL or DISABLED."
            )
        case AutoresetMode.DISABLED:
            discount = None
        case AutoresetMode.NORMAL | AutoresetMode.ENVPOOL:
            pass
        case _:
            raise ValueError(f"Unsupported autoreset mode: {autoreset_mode!r}.")

    env = EnvPoolGymAdapter(
        env_name=env_name,
        max_episode_steps=episode_length,
        num_envs=parallel,
        discount=discount,
        record_ori_obs=record_ori_obs,
        autoreset_next_step=autoreset_mode != AutoresetMode.NORMAL,
        **kwargs,
    )

    if autoreset_mode == AutoresetMode.DISABLED:
        env = OneEpisodeWrapper(env)

    return env


# Note: for env of Humanoid and HumanoidStandup, the action sapce is [-0.4, 0.4], we don't explicitly handle it. You need to manually squash the action space to [-1, 1] by using `ActionSquashWrapper`.
