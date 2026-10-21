from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from evorl.types import Action

from ..env import BaseEnv, EnvState
from ..space import Space
from .wrapper import Wrapper


class BraxMultiObjectiveRewardWrapper(Wrapper[Action, Space]):
    """Expose selected Brax metrics as a single-agent reward vector.

    Apply this wrapper after the adapter and before episode, action-repeat and
    vectorization wrappers. The wrapped environment must expose scalar metric
    arrays in `state.info.metrics`, as the Brax adapter does.
    Values are used unchanged: weighting, signs and time scaling retain
    their native metric definitions.

    Only the public `EnvState.reward` is replaced; native simulator state is
    preserved. Reset rewards are zero vectors. The final axis follows
    `reward_keys` order and is retained even when only one key is selected.

    Args:
        env: An unvectorized single-agent environment with per-step metrics.
        reward_keys: Nonempty sequence of scalar metric names to accumulate as
            reward objectives. These must exist on both reset and step.
    """

    def __init__(self, env: BaseEnv[Action, Space], reward_keys: Sequence[str]):
        super().__init__(env)
        if isinstance(reward_keys, str) or not reward_keys:
            raise ValueError("reward_keys must be a nonempty sequence of metric names.")
        self.reward_keys = tuple(reward_keys)

    def _reward_vector(self, state: EnvState) -> jax.Array:
        rewards = [jnp.asarray(state.info.metrics[name]) for name in self.reward_keys]
        if any(reward.ndim != 0 for reward in rewards):
            raise ValueError("Each reward metric must be scalar before vectorization.")
        return jnp.stack(rewards, axis=-1)

    def reset(self, key: jax.Array) -> EnvState:
        state = self.env.reset(key)
        return state.replace(reward=jnp.zeros_like(self._reward_vector(state)))

    def step(self, state: EnvState, action: Action) -> EnvState:
        state = self.env.step(state, action)
        return state.replace(reward=self._reward_vector(state))


class MujocoMultiObjectiveRewardWrapper(BraxMultiObjectiveRewardWrapper):
    """Expose selected MuJoCo Playground metrics as a reward vector.

    EvoRL's Playground adapter exposes the same `info.metrics` layout as the
    Brax adapter, so metric selection and reward conversion are inherited.
    Selected metrics retain their native signs, weights and time scaling.
    """


class RewardScaleWrapper[ActionT, SpaceT](Wrapper[ActionT, SpaceT]):
    """Scale the reward by a factor.

    Usage:
    - Use EpisodeWrapper(RewardScaleWrapper(env)) to get the scaled `info.episode_return`.
    - Use RewardScaleWrapper(EpisodeWrapper(env)) to get the original `info.episode_return`.
    """

    def __init__(self, env: BaseEnv[ActionT, SpaceT], reward_scale: float):
        super().__init__(env)
        self.reward_scale = reward_scale

    def reset(self, key: jax.Array) -> EnvState:
        state = self.env.reset(key)
        info = state.info.replace(ori_reward=state.reward)

        reward = jtu.tree_map(lambda r: r * self.reward_scale, state.reward)

        return state.replace(reward=reward, info=info)

    def step(self, state: EnvState, action: ActionT) -> EnvState:
        state = self.env.step(state, action)
        info = state.info.replace(ori_reward=state.reward)

        reward = jtu.tree_map(lambda r: r * self.reward_scale, state.reward)

        return state.replace(reward=reward, info=info)


class SparseRewardWrapper(Wrapper[Action, Space]):
    """Convert dense reward to sparse reward.

    Emit accumulated rewards every `sparse_length` steps and at episode end.
    Each emission clears the accumulator, including partial intervals at the
    terminal step. Scalar and vector rewards keep their original shape.

    Apply this wrapper after `EpisodeWrapper` (or `OneEpisodeWrapper`) and
    before vectorization and autoreset. This lets it observe time-limit `done`
    signals and emit pending rewards before the next episode starts.
    `info.episode_return` records the rewards seen by the inner episode wrapper,
    before sparse emission; discounted returns can differ after delaying rewards.
    """

    def __init__(self, env: BaseEnv[Action, Space], sparse_length: int):
        super().__init__(env)
        self.sparse_length = sparse_length

    def reset(self, key: jax.Array) -> EnvState:
        state = self.env.reset(key)

        state._internal.cum_count = jnp.zeros((), dtype=jnp.int32)
        state._internal.cum_reward = jtu.tree_map(
            lambda r: jnp.zeros_like(r), state.reward
        )

        return state

    def step(self, state: EnvState, action: Action) -> EnvState:
        state = self.env.step(state, action)

        cum_count = state._internal.cum_count
        cum_reward = state._internal.cum_reward

        cum_count = cum_count + 1
        cum_reward = jtu.tree_map(lambda x, y: x + y, cum_reward, state.reward)
        cond = jnp.logical_or(
            cum_count >= self.sparse_length, state.done.astype(jnp.bool)
        )

        reward = jtu.tree_map(
            lambda r: jnp.where(cond, r, jnp.zeros_like(r)), cum_reward
        )

        # reset cum_reward & cum_count
        cum_count = jnp.where(cond, jnp.zeros_like(cum_count), cum_count)
        cum_reward = jtu.tree_map(
            lambda r: jnp.where(cond, jnp.zeros_like(r), r), cum_reward
        )

        state = state.replace(
            reward=reward,
            _internal=state._internal.replace(
                cum_count=cum_count, cum_reward=cum_reward
            ),
        )

        return state
