"""Small native Jumanji environments for adapter and episode regressions."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.types import StepType, TimeStep, restart

from evorl.types import PyTreeDict


class CounterObservation(NamedTuple):
    value: jax.Array
    action_mask: jax.Array


class CounterEnv:
    """Make episode boundaries and extras observable without random termination."""

    action_spec = specs.DiscreteArray(2)
    observation_spec = specs.Spec(
        CounterObservation,
        value=specs.Array((1,), jnp.float32),
        action_mask=specs.BoundedArray((2,), bool, False, True),
    )
    reward_spec = specs.Array((), jnp.float32)

    def reset(self, key):
        state = PyTreeDict(steps=jnp.array(0), key=key)
        observation = CounterObservation(jnp.zeros(1), jnp.ones(2, dtype=bool))
        return state, restart(observation, extras={"counter": jnp.array(0)})

    def step(self, state, action):
        steps = state.steps + 1
        state = state.replace(steps=steps)
        return state, TimeStep(
            step_type=jnp.where(steps >= 3, StepType.LAST, StepType.MID),
            reward=jnp.array(1.0),
            discount=jnp.where(steps >= 3, 0.0, 1.0),
            observation=CounterObservation(
                jnp.full((1,), steps, dtype=jnp.float32), jnp.ones(2, dtype=bool)
            ),
            extras={"counter": steps},
        )


class StaggeredObservation(NamedTuple):
    value: jax.Array
    reset_marker: jax.Array
    action_mask: jax.Array


class StaggeredEnv:
    """End after two to four steps, with a reset-specific observation and key."""

    action_spec = specs.DiscreteArray(2)
    observation_spec = specs.Spec(
        StaggeredObservation,
        value=specs.Array((1,), jnp.float32),
        reset_marker=specs.BoundedArray((), jnp.float32, 0.0, 1.0),
        action_mask=specs.BoundedArray((2,), bool, False, True),
    )
    reward_spec = specs.Array((), jnp.float32)

    def reset(self, key):
        length_key, marker_key = jax.random.split(key)
        marker = jax.random.uniform(marker_key)
        state = PyTreeDict(
            steps=jnp.array(0),
            key=key,
            marker=marker,
            limit=jax.random.randint(length_key, (), 2, 5),
        )
        observation = StaggeredObservation(
            jnp.zeros(1), marker, jnp.array([True, False])
        )
        return state, restart(observation, extras={"counter": jnp.array(0)})

    def step(self, state, action):
        steps = state.steps + 1
        state = state.replace(steps=steps)
        done = steps >= state.limit
        observation = StaggeredObservation(
            jnp.full((1,), steps, dtype=jnp.float32),
            state.marker,
            jnp.arange(2) == steps % 2,
        )
        return state, TimeStep(
            step_type=jnp.where(done, StepType.LAST, StepType.MID),
            reward=steps.astype(jnp.float32),
            discount=1.0 - done,
            observation=observation,
            extras={"counter": steps},
        )
