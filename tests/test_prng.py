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
    rng_fold_in_axis,
    rng_key_batch_shape,
    rng_split,
    rng_split_by_shape,
    rng_split_like_tree,
    tree_has_nan,
)

from .utils import FakeEnv
from .helpers.sharding import compiled_collectives


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


def test_fold_in_axis_none_preserves_keys_and_tree(key):
    keys = {"rollout": key, "learn": (jax.random.key(18),)}
    assert rng_fold_in_axis(keys, None) is keys
    actual = jax.jit(lambda keys: rng_fold_in_axis(keys, None))(keys)
    for actual_key, expected in zip(
        jax.tree.leaves(actual), jax.tree.leaves(keys), strict=True
    ):
        np.testing.assert_array_equal(
            jax.random.key_data(actual_key), jax.random.key_data(expected)
        )
    with pytest.raises(TypeError):
        rng_fold_in_axis(keys)
    with pytest.raises(NameError, match="DP"):
        rng_fold_in_axis(key, "DP")


@pytest.mark.parametrize("nested", [False, True])
def test_fold_in_axis_matches_original_streams_without_communication(dp_mesh, nested):
    def local_step(key):
        root, rollout, learn = jax.random.split(key, 3)
        local = rng_fold_in_axis({"rollout": rollout, "learn": (learn,)}, "DP")
        return root, jax.tree.map(lambda k: jax.random.key_data(k)[None], local)

    mapped = jax.shard_map(
        local_step, in_specs=jax.P(), out_specs=(jax.P(), jax.P("DP"))
    )
    keys = jax.random.key(19)
    if nested:
        keys = jax.random.split(keys, (2, 3))
        mapped = jax.vmap(jax.vmap(mapped))
    compiled = jax.jit(mapped).lower(keys).compile()
    roots, actual = compiled(keys)

    def expected_step(key):
        root, rollout, learn = jax.random.split(key, 3)
        # Reproduce the former inline implementation directly for every rank.
        ranks = jnp.arange(dp_mesh.size, dtype=jnp.uint32)
        expected = jax.tree.map(
            lambda k: jax.random.key_data(
                jax.vmap(jax.random.fold_in, (None, 0))(k, ranks)
            ),
            {"rollout": rollout, "learn": (learn,)},
        )
        return root, expected

    if nested:
        expected_step = jax.vmap(jax.vmap(expected_step))
    expected_roots, expected = expected_step(keys)
    np.testing.assert_array_equal(
        jax.random.key_data(roots), jax.random.key_data(expected_roots)
    )
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
        assert np.all(
            np.any(np.asarray(a)[..., 0, :] != np.asarray(a)[..., 1, :], axis=-1)
        )
    assert not compiled_collectives(compiled.as_text())


@pytest.mark.four_devices
def test_fold_in_axis_uses_only_the_requested_axis(mesh_factory, key):
    mesh = mesh_factory((2, 2), ("DP", "POP"))
    with jax.set_mesh(mesh):

        @jax.shard_map(in_specs=jax.P(), out_specs=(jax.P("DP"), jax.P("POP")))
        def local(key):
            return (
                jax.random.key_data(rng_fold_in_axis(key, "DP"))[None],
                jax.random.key_data(rng_fold_in_axis(key, "POP"))[None],
            )

        compiled = jax.jit(local).lower(key).compile()
        dp_keys, pop_keys = compiled(key)
    expected = jax.random.key_data(
        jax.vmap(jax.random.fold_in, (None, 0))(key, jnp.arange(2, dtype=jnp.uint32))
    )
    np.testing.assert_array_equal(dp_keys, expected)
    np.testing.assert_array_equal(pop_keys, expected)
    assert not compiled_collectives(compiled.as_text())


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
def test_multiobjective_evaluator_maps_typed_key_batch_axes(batch_shape):
    from evorl.evaluators import Evaluator
    from evorl.envs.wrappers import BraxMultiObjectiveRewardWrapper

    class MetricEnv(FakeEnv):
        def reset(self, key):
            state = super().reset(key)
            state.info.metrics = PyTreeDict(value=state.reward)
            return state

        def step(self, state, action):
            state = super().step(state, action)
            return state.replace(
                info=state.info.replace(metrics=PyTreeDict(value=state.reward))
            )

    env = wrappers.VmapWrapper(
        wrappers.OneEpisodeWrapper(
            BraxMultiObjectiveRewardWrapper(
                MetricEnv(jnp.ones(2), jnp.array([0.0, 1.0])), ("value",)
            ),
            episode_length=2,
        ),
        num_envs=2,
    )

    def action_fn(state, batch, key):
        return jnp.zeros(batch.obs.shape[:-1], dtype=jnp.int32), PyTreeDict()

    evaluator = Evaluator(
        env=env,
        action_fn=action_fn,
        max_episode_steps=2,
    )
    keys = jax.random.split(jax.random.key(0), batch_shape)
    evaluate = lambda k: evaluator.evaluate(None, k, 4)
    for _ in batch_shape:
        evaluate = jax.vmap(evaluate)
    metrics = jax.jit(evaluate)(keys)
    assert metrics.episode_return.shape == (*batch_shape, 4, 1)
    np.testing.assert_array_equal(metrics.episode_return, 2.0)
    np.testing.assert_array_equal(metrics.episode_length, 2)
