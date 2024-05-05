import chex
import jax
from ..space import Box
from ..env import Env, EnvState
from .wrapper import Wrapper
from evorl.types import Action, Observation


class ObsFlattenWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        self.obs_ndim = len(env.obs_space.shape)

    def step(self, state: EnvState, action: Action) -> EnvState:
        state = self.env.step(state, action)
        start_idx = state.obs.ndim - self.obs_ndim
        state = state.replace(
            obs=jax.lax.collapse(state.obs, start_idx)
        )

        if 'last_obs' in state.info:
            state.info.last_obs = jax.lax.collapse(
                state.info.last_obs, start_idx
            )

        return state
