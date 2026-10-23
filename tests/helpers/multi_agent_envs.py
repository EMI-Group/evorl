"""Deterministic joint tasks with heterogeneous agents and staggered deaths."""

import jax.numpy as jnp

from evorl.envs import Box, Discrete, EnvState, MultiAgentEnv, SpaceContainer
from evorl.types import PyTreeDict


class ToyMultiAgentEnv(MultiAgentEnv):
    """Two agents; one terminates early, the other reaches the episode limit."""

    def __init__(self, continuous=False, global_obs=True):
        self.continuous = continuous
        self.global_obs = global_obs

    @property
    def agents(self):
        return ["first", "second"]

    @property
    def action_space(self):
        if self.continuous:
            return dict(
                first=Box(low=-2 * jnp.ones(2), high=2 * jnp.ones(2)),
                second=Box(low=jnp.zeros(1), high=jnp.ones(1)),
            )
        return dict(first=Discrete(n=2), second=Discrete(n=3))

    @property
    def obs_space(self):
        spaces = dict(
            first=Box(low=-jnp.ones(3) * 100, high=jnp.ones(3) * 100),
            second=Box(low=-jnp.ones(2) * 100, high=jnp.ones(2) * 100),
        )
        if not self.continuous:
            spaces = {
                name: SpaceContainer(
                    spaces=PyTreeDict(
                        observation=space,
                        action_mask=Box(
                            low=jnp.zeros(self.action_space[name].n, jnp.bool_),
                            high=jnp.ones(self.action_space[name].n, jnp.bool_),
                        ),
                    )
                )
                for name, space in spaces.items()
            }
        return spaces

    def _observations(self, count):
        obs = PyTreeDict(
            first=jnp.array([count, 1, -1], jnp.float32),
            second=jnp.array([count, 2], jnp.float32),
        )
        if not self.continuous:
            obs = PyTreeDict(
                {
                    name: PyTreeDict(
                        observation=value,
                        action_mask=jnp.arange(self.action_space[name].n)
                        == count % self.action_space[name].n,
                    )
                    for name, value in obs.items()
                }
            )
        return obs

    def reset(self, key):
        zeros = PyTreeDict(
            first=jnp.float32(0), second=jnp.float32(0), __all__=jnp.float32(0)
        )
        return EnvState(
            env_state=jnp.int32(0),
            obs=self._observations(jnp.int32(0)),
            reward=PyTreeDict(first=jnp.float32(0), second=jnp.float32(0)),
            done=zeros,
            info=PyTreeDict(global_obs=jnp.array([0, 10], jnp.float32))
            if self.global_obs
            else PyTreeDict(),
            _internal=PyTreeDict(termination=zeros, truncation=zeros),
        )

    def step(self, state, action):
        count = state.env_state + 1
        done = PyTreeDict(
            first=(count >= 2).astype(jnp.float32),
            second=(count >= 4).astype(jnp.float32),
            __all__=(count >= 4).astype(jnp.float32),
        )
        info = (
            state.info.replace(global_obs=jnp.array([count, 10], jnp.float32))
            if self.global_obs
            else state.info
        )
        return state.replace(
            env_state=count,
            obs=self._observations(count),
            done=done,
            reward=PyTreeDict(first=jnp.float32(1), second=jnp.float32(-2)),
            info=info,
            _internal=state._internal.replace(termination=done),
        )
