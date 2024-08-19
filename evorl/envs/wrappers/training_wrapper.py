from enum import Enum

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.utils.jax_utils import vmap_rng_split

from ..env import Env, EnvState
from .wrapper import Wrapper


class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end.

    This is the same as brax's EpisodeWrapper, and add some new fields in transition.info.
    Including:
        - steps: the current step count of the episode
        - trunction: whether the episode is truncated
        - termination: whether the episode is terminated
        - last_obs: the next observation without autoreset

    args:
        env: the wrapped env should be a single un-vectorized environment.
        episode_length: the maxiumum length of each episode for truncation
        action_repeat: the number of times to repeat each action
        record_episode_return: whether to record the return of each episode
    """

    def __init__(
        self,
        env: Env,
        episode_length: int,
        record_episode_return: bool = False,
        discount: float = 1.0,
    ):
        super().__init__(env)
        self.episode_length = episode_length
        self.record_episode_return = record_episode_return
        self.discount = discount

    def reset(self, key: chex.PRNGKey) -> EnvState:
        state = self.env.reset(key)

        state.info.steps = jnp.zeros((), dtype=jnp.int32)
        state.info.termination = jnp.zeros(())
        state.info.truncation = jnp.zeros(())
        state.info.last_obs = jnp.zeros_like(state.obs)
        if self.record_episode_return:
            state.info.episode_return = jnp.zeros(())

        return state

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        return self._step(state, action)

    def _step(self, state: EnvState, action: jax.Array) -> EnvState:
        prev_done = state.done
        # reset steps when prev episode is done(truncation or termination)
        steps = state.info.steps * (1 - prev_done).astype(jnp.int32)

        if self.record_episode_return:
            # reset the episode_return when the episode is done
            episode_return = state.info.episode_return * (1 - prev_done)

        # ==============
        state = self.env.step(state, action)

        # ============== post update =========
        steps = steps + 1

        done = jnp.where(
            steps >= self.episode_length, jnp.ones_like(state.done), state.done
        )

        state.info.steps = steps
        state.info.termination = state.done
        # Note: here we also consider the case:
        # when termination and truncation are both happened
        # at the last step, we set truncation=0
        state.info.truncation = jnp.where(
            steps >= self.episode_length, 1 - state.done, jnp.zeros_like(state.done)
        )
        # the real next_obs at the end of episodes, where
        # state.obs could be changed in VmapAutoResetWrapper
        # by the next episode's inital state
        state.info.last_obs = state.obs

        if self.record_episode_return:
            if self.discount == 1.0:  # a shortcut for discount=1.0
                episode_return += state.reward
            else:
                episode_return += jnp.power(self.discount, steps - 1) * state.reward
            state.info.episode_return = episode_return

        return state.replace(done=done)


class OneEpisodeWrapper(EpisodeWrapper):
    """Maintains episode step count and sets done at episode end.

    When call step() after the env is done, stop simulation and
    directly return last state.

    args:
        env: the wrapped env should be a single un-vectorized environment.

    """

    def __init__(self, env: Env, episode_length: int):
        super().__init__(env, episode_length, False)

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        return jax.lax.cond(
            state.done, lambda state, action: state.replace(), self._step, state, action
        )


class VmapWrapper(Wrapper):
    """
    Vectorizes Brax env.
    """

    def __init__(self, env: Env, num_envs: int = 1, vmap_step: bool = False):
        super().__init__(env)
        self.num_envs = num_envs
        self.vmap_step = vmap_step

    def reset(self, key: chex.PRNGKey) -> EnvState:
        """
        Args:
            key: support batched keys [B,2] or single key [2]
        """
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
    def __init__(self, env: Env, num_envs: int = 1):
        super().__init__(env)
        self.num_envs = num_envs

    def reset(self, key: chex.PRNGKey) -> EnvState:
        """
        Args:
            key: support batched keys [B,2] or single key [2]
        """
        if key.ndim <= 1:
            key = jax.random.split(key, self.num_envs)
        else:
            chex.assert_shape(
                key,
                (self.num_envs, 2),
                custom_message=f"Batched key shape {key.shape} must match num_envs: {self.num_envs}",
            )

        reset_key, key = vmap_rng_split(key)
        state = jax.vmap(self.env.reset)(key)
        state.extra.reset_key = reset_key  # for autoreset

        return state

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        state = jax.vmap(self.env.step)(state, action)

        # Map heterogeneous computation (non-parallelizable).
        # This avoids lax.cond becoming lax.select in vmap
        state = jax.lax.map(self._maybe_reset, state)

        return state

    def _auto_reset(self, state: EnvState) -> EnvState:
        """Reset the state and overwrite `timestep.observation` with the reset observation
        if the episode has terminated.

            Note: run on single env
        """
        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        new_key, reset_key = jax.random.split(state.extra.reset_key)
        reset_state = self.env.reset(reset_key)

        state = state.replace(
            env_state=reset_state.env_state,
            obs=reset_state.obs,
        )

        state.extra.reset_key = new_key

        return state

    def _maybe_reset(self, state: EnvState) -> EnvState:
        """Overwrite the state and timestep appropriately if the episode terminates.

        Note: run on single env
        """

        return jax.lax.cond(
            state.done,
            self._auto_reset,
            lambda state: state,
            state,
        )


class FastVmapAutoResetWrapper(Wrapper):
    """
    Brax-style AutoReset: no randomness in reset.
    This wrapper is more efficient than VmapAutoResetWrapper.
    """

    def __init__(self, env: Env, num_envs: int = 1):
        super().__init__(env)
        self.num_envs = num_envs

    def reset(self, key: chex.PRNGKey) -> EnvState:
        """
        Args:
            key: support batched keys [B,2] or single key [2]
        """
        if key.ndim <= 1:
            key = jax.random.split(key, self.num_envs)
        else:
            chex.assert_shape(
                key,
                (self.num_envs, 2),
                custom_message=f"Batched key shape {key.shape} must match num_envs: {self.num_envs}",
            )

        state = jax.vmap(self.env.reset)(key)
        state.extra.first_env_state = state.env_state
        state.extra.first_obs = state.obs

        return state

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        state = jax.vmap(self.env.step)(state, action)

        def where_done(x, y):
            done = state.done
            if done.ndim > 0:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        env_state = jax.tree_map(
            where_done, state.extra.first_env_state, state.env_state
        )
        obs = where_done(state.extra.first_obs, state.obs)

        return state.replace(env_state=env_state, obs=obs)


class VmapEnvPoolAutoResetWrapper(Wrapper):
    """
    EnvPool style AutoReset: an additional reset step after the episode ends.
    """

    def __init__(self, env: Env, num_envs: int = 1):
        super().__init__(env)
        self.num_envs = num_envs

    def reset(self, key: chex.PRNGKey) -> EnvState:
        """
        Args:
            key: support batched keys [B,2] or single key [2]
        """
        if key.ndim <= 1:
            key = jax.random.split(key, self.num_envs)
        else:
            chex.assert_shape(
                key,
                (self.num_envs, 2),
                custom_message=f"Batched key shape {key.shape} must match num_envs: {self.num_envs}",
            )

        reset_key, key = vmap_rng_split(key)
        state = jax.vmap(self.env.reset)(key)
        state.info.autoreset = jnp.zeros_like(state.done)  # for autoreset flag
        state.extra.reset_key = reset_key  # for autoreset

        return state

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        def _step(state: EnvState, action: jax.Array) -> EnvState:
            # on single env
            new_state = self.env.step(state, action)
            new_state.info.autoreset = state.done
            return new_state

        autoreset = state.done

        def _where_done(x, y):
            done = autoreset
            if done.ndim > 0:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        reset_state = self.reset(state.extra.reset_key)
        new_state = new_state = jax.vmap(self.env.step)(state, action)
        new_state.info.autoreset = autoreset

        # Map heterogeneous computation (non-parallelizable).
        # This avoids lax.cond becoming lax.select in vmap
        state = jtu.tree_map(
            _where_done,
            reset_state,
            new_state,
        )

        return state


class AutoresetMode(Enum):
    NORMAL = "normal"
    FAST = "fast"
    DISABLED = "disabled"
    ENVPOOL = "envpool"
