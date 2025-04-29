import chex
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

        return state.replace(reward=state.reward * self.reward_scale, info=info)

    def step(self, state: EnvState, action: Action) -> EnvState:
        state = self.env.step(state, action)
        info = state.info.replace(ori_reward=state.reward)

        return state.replace(reward=state.reward * self.reward_scale, info=info)
