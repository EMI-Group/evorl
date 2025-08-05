import chex
import jax
import jax.tree_util as jtu
from jax import numpy as jnp

from evorl.utils.jax_utils import rng_split

from ..env import Env, EnvState
from .wrapper import Wrapper


class EpisodeWrapper(Wrapper):
    """Multi-Agent version of the [EpisodeWrapper](#training_wrapper.EpisodeWrapper)."""

    def __init__(self, env: Env, episode_length: int):
        super().__init__(env)
        self.episode_length = episode_length

    def reset(self, key: chex.PRNGKey) -> EnvState:
        state = self.env.reset(key)

        state.info.steps = jnp.zeros((), dtype=jnp.int32)
        state.info.termination = jnp.zeros(())
        state.info.truncation = jnp.zeros(())
        state.info.ori_obs = jtu.tree_map(lambda x: jnp.zeros_like(x), state.obs)

        return state

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        return self._step(state, action)

    def _step(self, state: EnvState, action: jax.Array) -> EnvState:
        state = self.env.step(state, action)

        termination = state.done["__all__"]
        steps = state.info.steps * (1 - termination).astype(jnp.int32) + 1
        done = jnp.where(
            steps >= self.episode_length, jnp.ones_like(termination), termination
        )

        agents_done = jtu.tree_map(
            lambda x: jnp.where(done, x, jnp.ones_like(x)), state.done
        )

        state.info.steps = steps
        state.info.termination = termination
        state.info.truncation = jnp.where(
            steps >= self.episode_length, 1 - termination, jnp.zeros_like(termination)
        )
        # the real next_obs at the end of episodes, where
        # state.obs could be changed in VmapAutoResetWrapper
        state.info.ori_obs = state.obs

        return state.replace(done=agents_done)


class OneEpisodeWrapper(EpisodeWrapper):
    """Multi-Agent version of the [OneEpisodeWrapper](#training_wrapper.OneEpisodeWrapper)."""

    def __init__(self, env: Env, episode_length: int):
        super().__init__(env, episode_length)

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        return jax.lax.cond(
            state.done["__all__"],
            lambda state, action: state.replace(),
            self._step,
            state,
            action,
        )


class VmapWrapper(Wrapper):
    """Multi-Agent version of the [VmapWrapper](#training_wrapper.VmapWrapper)."""

    def __init__(self, env: Env, num_envs: int = 1, vmap_step: bool = False):
        super().__init__(env)
        self.num_envs = num_envs
        self.vmap_step = vmap_step

    def reset(self, key: chex.PRNGKey) -> EnvState:
        if key.ndim <= 1:
            key = jax.random.split(key, self.num_envs)
        else:
            chex.assert_shape(
                key,
                (self.num_envs, 2),
                custom_message=f"Batched key shape {key.shape} must match num_envs: {self.num_envs}",
            )

        return jax.vmap(self.env.reset)(key)

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        if self.vmap_step:
            return jax.vmap(self.env.step)(state, action)
        else:
            return jax.lax.map(lambda x: self.env.step(*x), (state, action))


class VmapAutoResetWrapper(Wrapper):
    """Multi-Agent version of the [VmapAutoResetWrapper](#training_wrapper.VmapAutoResetWrapper)."""

    def __init__(self, env: Env, num_envs: int = 1):
        super().__init__(env)
        self.num_envs = num_envs

    def reset(self, key: chex.PRNGKey) -> EnvState:
        if key.ndim <= 1:
            key = jax.random.split(key, self.num_envs)
        else:
            chex.assert_shape(
                key,
                (self.num_envs, 2),
                custom_message=f"Batched key shape {key.shape} must match num_envs: {self.num_envs}",
            )

        reset_key, key = rng_split(key)
        state = jax.vmap(self.env.reset)(key)
        state.info.reset_key = reset_key  # for autoreset

        return state

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        state = jax.vmap(self.env.step)(state, action)

        # Map heterogeneous computation (non-parallelizable).
        # This avoids lax.cond becoming lax.select in vmap
        state = jax.lax.map(self._maybe_reset, state)

        return state

    def _auto_reset(self, state: EnvState) -> EnvState:
        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        new_key, reset_key = jax.random.split(state.info.reset_key)
        reset_state = self.env.reset(reset_key)

        state = state.replace(
            env_state=reset_state.env_state,
            obs=reset_state.obs,
        )

        state.info.reset_key = new_key

        return state

    def _maybe_reset(self, state: EnvState) -> EnvState:
        return jax.lax.cond(
            state.done["__all__"],
            self._auto_reset,
            lambda state: state,
            state,
        )


class FastVmapAutoResetWrapper(Wrapper):
    """Multi-Agent version of the [FastVmapAutoResetWrapper](#training_wrapper.FastVmapAutoResetWrapper)."""

    def __init__(self, env: Env, num_envs: int = 1):
        super().__init__(env)
        self.num_envs = num_envs

    def reset(self, key: chex.PRNGKey) -> EnvState:
        if key.ndim <= 1:
            key = jax.random.split(key, self.num_envs)
        else:
            chex.assert_shape(
                key,
                (self.num_envs, 2),
                custom_message=f"Batched key shape {key.shape} must match num_envs: {self.num_envs}",
            )

        state = jax.vmap(self.env.reset)(key)
        state.info.first_env_state = state.env_state
        state.info.first_obs = state.obs

        return state

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        state = jax.vmap(self.env.step)(state, action)

        def where_done(x, y):
            done = state.done["__all__"]
            if done.ndim > 0:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        env_state = jtu.tree_map(
            where_done, state.info.first_env_state, state.env_state
        )
        obs = where_done(state.info.first_obs, state.obs)

        return state.replace(env_state=env_state, obs=obs)
