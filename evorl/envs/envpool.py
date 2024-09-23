from functools import partial

import chex
import envpool
import gym
import gym.spaces
import gymnasium
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from envpool.python.protocol import EnvPool
from jax.experimental import io_callback

from evorl.types import Action, PyTreeDict

from .env import Env, EnvAdapter, EnvState
from .space import Box, Discrete, Space
from .wrappers import Wrapper, AutoresetMode


def _to_jax(x):
    return jtu.tree_map(lambda x: jnp.asarray(x), x)


def _to_jax_spec(x):
    x = _to_jax(x)
    return jtu.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), x)


def _to_numpy(x):
    return jtu.tree_map(lambda x: np.asarray(x), x)


class GymAdapter(EnvAdapter):
    """
    Adapter for EnvPool to support gym(>=0.26.2) and gymnasium environments.
    TODO: multi-device support
    """

    def __init__(self, env: EnvPool):
        super().__init__(env)
        self.num_envs = env.config["num_envs"]

        reset_spec = _to_jax_spec(self.env.reset())

        dummy_action = self.env.action_space.sample()
        dummy_actions = np.broadcast_to(
            dummy_action, (self.num_envs,) + dummy_action.shape
        )
        step_spec = _to_jax_spec(self.env.step(dummy_actions))

        def _reset():
            return self.env.reset()

        def _step(action):
            return self.env.step(np.asarray(action))

        self._reset = partial(io_callback, _reset, reset_spec)
        self._step = partial(io_callback, _step, step_spec)

    def reset(self, key: chex.PRNGKey) -> EnvState:
        obs, info = _to_jax(self._reset())

        # Note: we drop the original info as they are not static

        info = PyTreeDict(
            termination=jnp.zeros((self.num_envs,)),
            truncation=jnp.zeros((self.num_envs,)),
            episode_return=jnp.zeros((self.num_envs,)),
            autoreset=jnp.zeros((self.num_envs,)),
        )

        return EnvState(
            env_state=None,
            obs=obs,
            reward=jnp.zeros((self.num_envs,)),
            done=jnp.zeros((self.num_envs,)),
            info=info,
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        episode_return = state.info.episode_return * (1 - state.done)

        obs, reward, termination, truncation, info = _to_jax(
            self._step(
                action,
            )
        )

        reward = reward.astype(jnp.float32)
        done = (jnp.logical_or(termination, truncation)).astype(jnp.float32)

        # when autoreset happens (indicated by prev_done)
        # we add a new field `autoreset` to mark invalid transition for the additional reset() step in envpool.
        # use it in q-learning based algorithms
        info = state.info.replace(
            termination=termination.astype(jnp.float32),
            truncation=truncation.astype(jnp.float32),
            episode_return=episode_return + reward,
            autoreset=state.done,  # prev_done
        )

        return state.replace(obs=obs, reward=reward, done=done, info=info)

    @property
    def action_space(self) -> Space:
        return gym_space_to_evorl_space(self.env.action_space)

    @property
    def obs_space(self) -> Space:
        return gym_space_to_evorl_space(self.env.observation_space)


class OneEpisodeWrapper(Wrapper):
    """
    Vectorized one episode wrapper for evaluation.
    """

    def __init__(self, env: Env):
        super().__init__(env)

    def step(self, state: EnvState, action: Action) -> EnvState:
        """
        Note: could add extra CPU overhead
        """

        def where_done(x, y):
            done = state.done
            if done.ndim > 0:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
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


def creat_gym_env(
    env_name,
    gymnasium_env=True,
    episode_length: int = 1000,
    parallel: int = 1,
    autoreset_mode: AutoresetMode = AutoresetMode.ENVPOOL,
    **kwargs,
) -> GymAdapter:
    """
    Tips:
    1. unlike other jax-based env, most wrappers are handled in envpool.
    2. Don't use the env with vmap, eg: vmap(env.step), this could cause undefined behavior.
    """

    if autoreset_mode not in [AutoresetMode.ENVPOOL, AutoresetMode.DISABLED]:
        raise ValueError(
            "Only AutoresetMode.ENVPOOL and AutoresetMode.DISABLED are supported for envpool based env."
        )

    if gymnasium_env:
        env = envpool.make_gymnasium(
            env_name, num_envs=parallel, max_episode_steps=episode_length, **kwargs
        )
    else:
        env = envpool.make_gym(
            env_name, num_envs=parallel, max_episode_steps=episode_length, **kwargs
        )

    env = GymAdapter(env)

    if autoreset_mode == AutoresetMode.DISABLED:
        env = OneEpisodeWrapper(env)

    return env


# Note: for env of Humanoid and HumanoidStandup, the action sapce is [-0.4, 0.4], we don't explicitly handle it. You need to manually squash the action space to [-1, 1] by using `ActionSquashWrapper`.
