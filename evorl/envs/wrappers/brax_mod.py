from typing import Optional, Type


from brax.envs.base import Env, State, Wrapper
import jax
from jax import numpy as jnp

class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end.

    This is the same as brax's EpisodeWrapper, and add some new fields in state.info.

    args:
        env: the wrapped env should be a single un-vectorized environment.

    """

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info["termination"] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info["termination"] = state.done
        state.info['truncation'] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )

        state.info['steps'] = steps

        return state.replace(done=done)


class EpisodeWrapperV2(Wrapper):
    """Maintains episode step count and sets done at episode end.

    When call step() after the env is done, directly return last state.

    args:
        env: the wrapped env should be a single un-vectorized environment.

    """

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info["termination"] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state: State, action: jax.Array) -> State:
        return jax.lax.cond(
            state.done,
            self._normal_step,
            self._dummy_step,
            state, action
        )

    def _normal_step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info["termination"] = state.done
        state.info['truncation'] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )

        state.info['steps'] = steps

        return state.replace(done=done)

    def _dummy_step(self, state: State, action: jax.Array) -> State:
        return state.replace()


class VmapWrapper(Wrapper):
    """
        Vectorizes Brax env.

        Args:
            batch_size: mauanlly define #num envs to vectorize.
                if None, will be inferred from reset rng.shape[0].
    """

    def __init__(self, env: Env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)

    def step(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(state, action)
    
    def num_envs(self, state):
        return state.obs.shape[0]


class AutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        # add last_obs for PEB (when calc GAE)
        state.info['last_obs'] = jnp.zeros_like(state.obs)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(
                    done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        pipeline_state = jax.tree_map(
            where_done, state.info['first_pipeline_state'], state.pipeline_state
        )
        # the real next_obs at the end of an episode
        state.info['last_obs'] = state.obs
        obs = where_done(state.info['first_obs'], state.obs)

        return state.replace(pipeline_state=pipeline_state, obs=obs)


def has_wrapper(env: Env, wrapper_cls: Type) -> bool:
    """Check if env has a wrapper of type wrapper_cls."""
    while isinstance(env, Wrapper):
        if isinstance(env, wrapper_cls):
            return True
        env = env.env
    return False