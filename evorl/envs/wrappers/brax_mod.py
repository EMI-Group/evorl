from typing import Optional, Type, Dict


from brax.envs.base import Env, State, Wrapper
import jax
from jax import numpy as jnp
from flax import struct
import chex


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

    def reset(self, rng: chex.PRNGKey) -> State:
        state = self.env.reset(rng)
        state.info['steps'] = jnp.zeros((), dtype=jnp.int32)
        state.info["termination"] = jnp.zeros(())
        state.info['truncation'] = jnp.zeros(())
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

    When call step() after the env is done, stop simulation and 
    directly return last state.

    args:
        env: the wrapped env should be a single un-vectorized environment.

    """

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: chex.PRNGKey) -> State:
        state = self.env.reset(rng)
        state.info['steps'] = jnp.zeros((), dtype=jnp.int32)
        state.info["termination"] = jnp.zeros(())
        state.info['truncation'] = jnp.zeros(())
        return state

    def step(self, state: State, action: jax.Array) -> State:
        return jax.lax.cond(
            state.done,
            self._dummy_step,
            self._normal_step,
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


class EpisodeRecordWrapper(Wrapper):
    def __init__(self, env: Env, discount: float):
        super().__init__(env)
        self.discount = discount

    def reset(self, rng: chex.PRNGKey) -> State:
        state = self.env.reset(rng)
        state.info['episode_return'] = jnp.zeros(())
        return state

    def step(self, state: State, action: jax.Array) -> State:
        prev_done = state.info.get('autoreset', state.done)
        episode_return = state.info['episode_return'] * (1-prev_done)

        state = self.env.step(state, action)
        
        steps = state.info['steps']
        episode_return += jnp.power(self.discount, steps-1)*state.reward
        state.info['episode_return'] = episode_return
        return state


class VmapWrapper(Wrapper):
    """
        Vectorizes Brax env.

        Args:
            batch_size: mauanlly define #num envs to vectorize.
                if None, will be inferred from reset rng.shape[0].
    """

    def __init__(self, env: Env, num_envs: int = 1):
        super().__init__(env)
        self.num_envs = num_envs

    def reset(self, rng: chex.PRNGKey) -> State:
        rng = jax.random.split(rng, self.num_envs)
        return jax.vmap(self.env.reset)(rng)

    def step(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(state, action)

    def num_envs(self, state):
        return state.obs.shape[0]


class AutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: chex.PRNGKey) -> State:
        state = self.env.reset(rng)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        # add last_obs for PEB (when calc GAE)
        state.info['last_obs'] = jnp.zeros_like(state.obs)
        state.info["autoreset"] = jnp.zeros_like(state.done)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state.info['autoreset'] = state.done # keep the original done
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


# @struct.dataclass
# class EpisodeMetrics:
#     """Dataclass holding evaluation metrics for Brax.

#     Attributes:
#         episode_metrics: Aggregated episode metrics since the beginning of the
#           episode.
#         active_episodes: Boolean vector tracking which episodes are not done yet.
#         episode_steps: Integer vector tracking the number of steps in the episode.
#     """

#     episode_metrics: Dict[str, jax.Array]
#     active_episodes: jax.Array
#     episode_steps: jax.Array


# class EvalWrapper(Wrapper):
#     """Brax env with eval metrics."""

#     def reset(self, rng: jax.Array) -> State:
#         reset_state = self.env.reset(rng)
#         reset_state.metrics['reward'] = reset_state.reward
#         eval_metrics = EpisodeMetrics(
#             episode_metrics=jax.tree_util.tree_map(
#                 jnp.zeros_like, reset_state.metrics
#             ),
#             active_episodes=jnp.ones_like(reset_state.reward),
#             episode_steps=jnp.zeros_like(reset_state.reward),
#         )
#         reset_state.info['eval_metrics'] = eval_metrics
#         return reset_state

#     def step(self, state: State, action: jax.Array) -> State:
#         state_metrics = state.info['eval_metrics']
#         if not isinstance(state_metrics, EpisodeMetrics):
#             raise ValueError(
#                 f'Incorrect type for state_metrics: {type(state_metrics)}'
#             )
#         del state.info['eval_metrics']
#         nstate = self.env.step(state, action)
#         nstate.metrics['reward'] = nstate.reward
#         episode_steps = jnp.where(
#             state_metrics.active_episodes,
#             nstate.info['steps'],
#             state_metrics.episode_steps,
#         )
#         episode_metrics = jax.tree_util.tree_map(
#             lambda a, b: a + b * state_metrics.active_episodes,
#             state_metrics.episode_metrics,
#             nstate.metrics,
#         )
#         active_episodes = state_metrics.active_episodes * (1 - nstate.done)

#         eval_metrics = EpisodeMetrics(
#             episode_metrics=episode_metrics,
#             active_episodes=active_episodes,
#             episode_steps=episode_steps,
#         )
#         nstate.info['eval_metrics'] = eval_metrics
#         return nstate


def has_wrapper(env: Env, wrapper_cls: Type) -> bool:
    """Check if env has a wrapper of type wrapper_cls."""
    while isinstance(env, Wrapper):
        if isinstance(env, wrapper_cls):
            return True
        env = env.env
    return False


def get_wrapper(env: Env, wrapper_cls: Type) -> Optional[Env]:
    while isinstance(env, Wrapper):
        if isinstance(env, wrapper_cls):
            return env
        env = env.env
    return None
