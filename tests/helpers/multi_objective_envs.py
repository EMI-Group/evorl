"""Small vector-reward tasks with opposing action preferences."""

import jax.numpy as jnp

from evorl.envs import Box, Discrete, Env, EnvState


class ToyMultiObjectiveEnv(Env):
    """Two actions favor different reward coordinates; episodes last two steps."""

    def __init__(self, continuous=False, num_objectives=2):
        self.continuous = continuous
        self.num_objectives = num_objectives

    @property
    def obs_space(self):
        return Box(low=-10 * jnp.ones(3), high=10 * jnp.ones(3))

    @property
    def action_space(self):
        if self.continuous:
            return Box(low=jnp.array([-2.0, 0.0]), high=jnp.array([2.0, 3.0]))
        return Discrete(n=2)

    def reset(self, key):
        return EnvState(
            env_state=jnp.int32(0),
            obs=jnp.array([0.0, 1.0, -1.0]),
            reward=jnp.zeros(self.num_objectives),
            done=jnp.float32(0),
        )

    def step(self, state, action):
        count = state.env_state + 1
        if self.continuous:
            rewards = jnp.array(
                [
                    -((action[0] - 1) ** 2) - (action[1] - 1.5) ** 2,
                    -((action[0] + 1) ** 2) - (action[1] - 1.5) ** 2,
                    -(action[0] ** 2),
                ]
            )
        else:
            rewards = jnp.array(
                [
                    jnp.where(action == 0, 2.0, 0.0),
                    jnp.where(action == 1, 3.0, 0.0),
                    1.0,
                ]
            )
        return state.replace(
            env_state=count,
            obs=jnp.array([count, 1.0, -1.0]),
            reward=rewards[: self.num_objectives],
            done=(count >= 2).astype(jnp.float32),
        )
