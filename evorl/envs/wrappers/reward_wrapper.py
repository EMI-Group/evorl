import chex
from evorl.types import Action

from ..env import Env, EnvState
from .wrapper import Wrapper


class RewardScaleWrapper(Wrapper):
    """Scale the reward by a factor."""

    def __init__(self, env: Env, scale: float):
        super().__init__(env)
        self.scale = scale

    def reset(self, key: chex.PRNGKey) -> EnvState:
        state = self.env.reset(key)
        info = state.info.replace(ori_reward=state.reward)

        return state.replace(reward=state.reward * self.scale, info=info)

    def step(self, state: EnvState, action: Action) -> EnvState:
        nstate = self.env.step(state, action)
        info = nstate.info.replace(ori_reward=state.reward)

        return nstate.replace(reward=nstate.reward * self.scale, info=info)
