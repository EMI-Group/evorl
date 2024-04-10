import jax.numpy as jnp
from .env import EnvAdapter, EnvState

import chex
from evorl.types import Action
from .space import Space, Box, Discrete
from .utils import sort_dict
import jumanji
from jumanji.env import Environment as JumanjiEnv
from jumanji.specs import Spec, DiscreteArray, BoundedArray, Array

# Note: this is used for singel agent envs.
class JumanjiAdapter(EnvAdapter):
    def __init__(self, env: JumanjiEnv):
        super(JumanjiAdapter, self).__init__(env)
        self._action_sapce = jumanji_specs_to_evorl_space(env.action_spec)
        self._obs_space = jumanji_specs_to_evorl_space(env.observation_spec)

    def reset(self, key: chex.PRNGKey) -> EnvState:
        env_state, transition = self.env.reset(key)
        return EnvState(
            env_state=env_state,
            obs=transition.observation,
            reward=transition.reward,
            done=jnp.asarray(transition.last(), dtype=jnp.float32),
            info=sort_dict(transition.extras)
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        env_state, transition = self.env.step(state, action)

        state.info.update(transition.extras)

        return state.replace(
            env_state=env_state,
            obs=transition.observation,
            reward=transition.reward,
            done=jnp.asarray(transition.last(), dtype=jnp.float32),
        )

    @property
    def action_space(self) -> Space:
        return self._action_sapce

    @property
    def obs_space(self) -> Space:
        return self._obs_space

# TODO: multi-agent EnvAdapter


def jumanji_specs_to_evorl_space(spec):
    if isinstance(spec, DiscreteArray):
        return Discrete(n=spec.spec.num_values)
    elif isinstance(spec, BoundedArray):
        low = jnp.broadcast_to(spec.minimum, spec.shape)
        high = jnp.broadcast_to(spec.maximum, spec.shape)
        return Box(low=low, high=high)
    elif isinstance(spec, Array):
        high = jnp.full(spec.shape, 1e10)
        return Box(low=-high, high=high)
    else:
        raise NotImplementedError(f"Unsupported space type: {type(spec)}")
    # TODO: support nested Space for envs like Sudoku


def create_jumanji_env(env_name: str, **kwargs) -> JumanjiAdapter:
    env = jumanji.make(env_name, **kwargs)
    env = JumanjiAdapter(env)

    # TODO: action wrapper

    return env
