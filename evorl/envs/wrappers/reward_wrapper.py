import chex
import jax.numpy as jnp
import jax.tree_util as jtu
from evorl.types import Action

from ..env import Env, EnvState
from .wrapper import Wrapper


class RewardScaleWrapper(Wrapper):
    """Scale the reward by a factor.

    Usage:
    - Use EpisodeWrapper(RewardScaleWrapper(env)) to get the scaled `info.episode_return`.
    - Use RewardScaleWrapper(EpisodeWrapper(env)) to get the original `info.episode_return`.
    """

    def __init__(self, env: Env, reward_scale: float):
        super().__init__(env)
        self.reward_scale = reward_scale

    def reset(self, key: chex.PRNGKey) -> EnvState:
        state = self.env.reset(key)
        info = state.info.replace(ori_reward=state.reward)

        reward = jtu.tree_map(lambda r: r * self.reward_scale, state.reward)

        return state.replace(reward=reward, info=info)

    def step(self, state: EnvState, action: Action) -> EnvState:
        state = self.env.step(state, action)
        info = state.info.replace(ori_reward=state.reward)

        reward = jtu.tree_map(lambda r: r * self.reward_scale, state.reward)

        return state.replace(reward=reward, info=info)


class SparseRewardWrapper(Wrapper):
    """Convert dense reward to sparse reward.

    The dense rewards become: 0, 0, ..., sum(rewards), 0, 0, ..., sum(rewards)
    """

    def __init__(self, env: Env, sparse_length: int):
        super().__init__(env)
        self.sparse_length = sparse_length

    def reset(self, key: chex.PRNGKey) -> EnvState:
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
