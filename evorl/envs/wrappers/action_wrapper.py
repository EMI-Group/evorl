import chex
from ..space import Box
from ..env import Env, EnvState
from .wrapper import Wrapper
from evorl.types import Action


class ActionSquashWrapper(Wrapper):
    """
        Convert continuous action space from [-1, 1] to [low, high]
    """

    def __init__(self, env: Env):
        super().__init__(env)

        # TODO: support pytree action space
        action_space = self.env.action_space
        assert isinstance(action_space, Box), "Only support Box action_space"

        self.scale = (action_space.high - action_space.low)*0.5
        self.bias = (action_space.high + action_space.low)*0.5

    def step(self, state: EnvState, action: Action) -> EnvState:
        squashed_action = self.scale*action + self.bias
        return self.env.step(state, squashed_action)
