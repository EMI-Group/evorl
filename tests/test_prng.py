"""Typed-key regressions for random helpers, environments and operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.agent import RandomAgent
from evorl.ec.operators import ERLMutation, MLPCrossover, MLPMutation
from evorl.envs import Discrete
from evorl.envs.wrappers import training_wrapper as wrappers
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.jax_utils import (
    ensure_batched_key,
    rng_key_batch_shape,
    rng_split,
    rng_split_by_shape,
    rng_split_like_tree,
    tree_has_nan,
)

from .utils import FakeEnv


@pytest.fixture
def key():
    return jax.random.key(17)


@pytest.mark.parametrize("impl", ["threefry2x32", "rbg"])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_split_preserves_key_batch_axes(impl, batch_shape):
    key = jax.random.key(17, impl=impl)
    keys = rng_split_by_shape(key, batch_shape)
    actual = jax.jit(lambda k: rng_split(k, 4))(keys)
    expected = jax.random.split
    for _ in batch_shape:
        expected = jax.vmap(expected, in_axes=(0, None), out_axes=1)
    np.testing.assert_array_equal(
        jax.random.key_data(actual), jax.random.key_data(expected(keys, 4))
    )
    assert rng_key_batch_shape(actual) == (4, *batch_shape)
    leaves = rng_split_like_tree(key, {"a": 0, "b": (1, 2)})
    assert len(jax.tree.leaves(leaves)) == 3


def test_batched_key_validation(key):
    keys = ensure_batched_key(key, 3)
    np.testing.assert_array_equal(
        jax.random.key_data(ensure_batched_key(keys, 3)),
        jax.random.key_data(keys),
    )
    with pytest.raises(ValueError, match="batch size"):
        ensure_batched_key(keys, 2)


@pytest.mark.parametrize("scalar_obs", [False, True])
def test_random_agent_samples_all_batch_axes(key, scalar_obs):
    agent = RandomAgent()
    obs_space = (
        Discrete(n=5) if scalar_obs else FakeEnv(jnp.ones(2), jnp.zeros(2)).obs_space
    )
    action_space = Discrete(n=7)
    state = agent.init(obs_space, action_space, key)
    obs = jnp.zeros((2, 3, *obs_space.shape))
    actions, _ = jax.jit(agent.compute_actions)(state, SampleBatch(obs=obs), key)
    expected = jax.vmap(jax.vmap(action_space.sample))(jax.random.split(key, (2, 3)))
    np.testing.assert_array_equal(actions, expected)
    assert actions.shape == (2, 3)


@pytest.mark.parametrize(
    "wrapper",
    [
        wrappers.VmapWrapper,
        wrappers.VmapAutoResetWrapper,
        wrappers.FastVmapAutoResetWrapper,
        wrappers.VmapEnvPoolAutoResetWrapper,
    ],
)
@pytest.mark.parametrize("prebatched", [False, True])
def test_vector_wrappers_reset_and_step(key, wrapper, prebatched):
    env = wrapper(FakeEnv(jnp.ones(2), jnp.ones(2)), num_envs=3)
    keys = jax.random.split(key, 3) if prebatched else key
    state = jax.jit(env.reset)(keys)
    for _ in range(3):
        state = jax.jit(env.step)(state, jnp.zeros(3, dtype=jnp.int32))
    assert state.obs.shape == (3, 7)
    assert rng_key_batch_shape(state.env_state.key) == (3,)


class MultiAgentFakeEnv(FakeEnv):
    def reset(self, key):
        state = super().reset(key)
        return state.replace(
            obs=PyTreeDict(agent=state.obs),
            reward=PyTreeDict(agent=state.reward),
            done=PyTreeDict(agent=state.done, __all__=state.done),
        )

    def step(self, state, action):
        return state.replace(
            env_state=state.env_state.replace(i=state.env_state.i + 1),
            done=PyTreeDict(agent=jnp.ones(()), __all__=jnp.ones(())),
        )


@pytest.mark.parametrize(
    "wrapper",
    [
        wrappers.VmapWrapper,
        wrappers.VmapAutoResetWrapper,
        wrappers.FastVmapAutoResetWrapper,
    ],
)
def test_multi_agent_vector_wrappers_with_batched_keys(key, wrapper):
    env = wrapper(MultiAgentFakeEnv(jnp.ones(2), jnp.ones(2)), num_envs=3)
    state = jax.jit(env.reset)(jax.random.split(key, 3))
    state = jax.jit(env.step)(state, PyTreeDict(agent=jnp.zeros(3)))
    assert state.obs.agent.shape == (3, 7)
    assert rng_key_batch_shape(state.env_state.key) == (3,)


@pytest.mark.parametrize("operator", [MLPMutation, ERLMutation, MLPCrossover])
def test_evolution_operators_accept_prebatched_keys(key, operator):
    population = {"kernel": jnp.arange(48, dtype=jnp.float32).reshape(4, 3, 4)}
    op = operator()
    batch_size = 2 if operator is MLPCrossover else 4
    expected = jax.jit(op)(population, key)
    actual = jax.jit(op)(population, jax.random.split(key, batch_size))
    np.testing.assert_array_equal(actual["kernel"], expected["kernel"])


def test_nan_check_accepts_typed_key_state():
    result = jax.jit(tree_has_nan)({"key": jax.random.key(0), "value": jnp.ones(2)})
    assert not result["key"]
    assert not result["value"]


def test_gymnasium_callback_accepts_typed_keys(monkeypatch):
    import gymnasium
    from evorl.envs.gymnasium import GymnasiumAdapter

    instances = []

    def make_vec(env_name, num_envs, max_episode_steps, **kwargs):
        env = gymnasium.vector.SyncVectorEnv(
            [lambda: gymnasium.make(env_name, max_episode_steps=max_episode_steps)]
            * num_envs
        )
        instances.append(env)
        return env

    monkeypatch.setattr(gymnasium, "make_vec", make_vec)
    try:
        env = GymnasiumAdapter("CartPole-v1", max_episode_steps=5, num_envs=2)
        state = jax.jit(env.reset)(jax.random.key(0))
        assert state.obs.shape == (2, 4)
        states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(1), 3))
        assert states.obs.shape == (3, 2, 4)
    finally:
        for env in instances:
            env.close()


@pytest.mark.parametrize("batch_shape", [(3,), (2, 3)])
def test_multiobjective_evaluator_maps_typed_key_batch_axes(monkeypatch, batch_shape):
    from evorl.evaluators import mo_brax_evaluator

    class MetricEnv(FakeEnv):
        def reset(self, key):
            state = super().reset(key)
            state.info.metrics = PyTreeDict()
            return state

    monkeypatch.setattr(mo_brax_evaluator, "BraxAdapter", MetricEnv)
    env = wrappers.VmapWrapper(
        MetricEnv(jnp.ones(2), jnp.array([0.0, 1.0])), num_envs=2
    )

    def action_fn(state, batch, key):
        return jnp.zeros(batch.obs.shape[:-1], dtype=jnp.int32), PyTreeDict()

    evaluator = mo_brax_evaluator.BraxEvaluator(
        env=env,
        action_fn=action_fn,
        max_episode_steps=2,
    )
    keys = jax.random.split(jax.random.key(0), batch_shape)
    metrics = jax.jit(lambda k: evaluator.evaluate(None, k, 4))(keys)
    assert metrics.reward.shape == (*batch_shape, 4)
    np.testing.assert_array_equal(metrics.reward, 2.0)
    np.testing.assert_array_equal(metrics.episode_lengths, 2)
