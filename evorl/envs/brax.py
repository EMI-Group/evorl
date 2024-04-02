import jax
import jax.numpy as jnp

from flax import struct
import chex
from .env import EnvAdapter, EnvState
from .space import Space, Box
from .utils import sort_dict
from evorl.types import Action, PyTreeDict
from brax.envs import (
    Env as BraxEnv,
    get_environment
)



class BraxAdapter(EnvAdapter):
    def __init__(self, env: BraxEnv):
        super(BraxAdapter, self).__init__(env)

        action_spec = jnp.asarray(
            env.sys.actuator.ctrl_range, dtype=jnp.float32)
        self._action_sapce = Box(low=action_spec[:, 0], high=action_spec[:, 1])

        obs_spec = jnp.full((env.observation_size,),
                            jnp.inf, dtype=jnp.float32)
        self._obs_space = Box(low=-obs_spec, high=obs_spec)

    def reset(self, key: chex.PRNGKey) -> EnvState:
        brax_state = self.env.reset(key)

        info = PyTreeDict(sort_dict(brax_state.info))
        info.metrics = PyTreeDict(sort_dict(brax_state.metrics))

        return EnvState(
            env_state=brax_state,
            obs=brax_state.obs,
            reward=brax_state.reward,
            done=brax_state.done,
            info=info
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        brax_state = self.env.step(state.env_state, action)

        state.info.update(brax_state.info)
        state.info.metrics = brax_state.metrics

        return state.replace(
            env_state=brax_state,
            obs=brax_state.obs,
            reward=brax_state.reward,
            done=brax_state.done,
        )


    @property
    def action_space(self) -> Space:
        return self._action_sapce

    @property
    def obs_space(self) -> Space:
        return self._obs_space


def create_brax_env(env_name: str, **kwargs) -> BraxAdapter:
    """
        Args:
            Autoreset: When use envs for RL training, set autoreset=True. When use envs for evaluation, set autoreset=False.
            discount: discount factor for episode return calculation. The episode returns are Only recorded when autoreset=True.
    """

    env = get_environment(env_name, **kwargs)
    env = BraxAdapter(env)

    return env
