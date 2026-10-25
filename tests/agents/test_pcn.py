"""PCN hindsight supervision, trajectory priorities and command sampling."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.algorithms.multi_objective.pcn import (
    PCNAgent,
    add_episodes,
    episode_priorities,
    episode_returns,
    sample_commands,
    sample_episode_batch,
)
from evorl.envs import Box, Discrete
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils import running_statistics


def archive(returns, lengths=None):
    """Make identifiable short trajectories, poisoning their unused padding."""
    returns = jnp.asarray(returns, jnp.float32)
    count, objectives = returns.shape
    lengths = jnp.asarray(lengths if lengths is not None else [2] * count, jnp.int32)
    observations = jnp.broadcast_to(
        jnp.arange(count)[:, None, None] * 10 + jnp.arange(3)[None, :, None],
        (count, 3, 1),
    )
    valid = jnp.arange(3)[None] < lengths[:, None]
    return PyTreeDict(
        obs=jnp.where(valid[..., None], observations, jnp.nan),
        actions=jnp.broadcast_to(jnp.arange(count)[:, None], (count, 3)),
        returns=jnp.broadcast_to(returns[:, None], (count, 3, objectives)),
        lengths=lengths,
    )


@pytest.mark.parametrize("discount", [1.0, 0.5])
def test_returns_relabel_only_real_transitions(discount, jit):
    rewards = np.array(
        [[[1, -1], [3, -1], [np.nan, np.nan]], [[2, 4], [4, 6], [8, 10]]], np.float32
    )
    valid = np.array([[True, True, False], [True, True, True]])
    expected = np.zeros_like(rewards)
    for episode in range(2):
        total = np.zeros(2)
        for t in reversed(range(3)):
            total = (
                rewards[episode, t] + discount * total
                if valid[episode, t]
                else np.zeros(2)
            )
            expected[episode, t] = total
    fn = jax.jit(episode_returns) if jit else episode_returns
    actual = fn(jnp.asarray(rewards), jnp.asarray(valid), discount)
    np.testing.assert_allclose(actual, expected)


def test_priorities_keep_fronts_penalize_duplicates_and_ignore_empty_slots():
    returns = jnp.array(
        [[5, 0], [0, 5], [2, 2], [1, 1], [5, 0], [100, 100]], jnp.float32
    )
    valid = jnp.array([True, True, True, True, True, False])
    scores, front = jax.jit(episode_priorities)(returns, valid)
    np.testing.assert_array_equal(front, [True, True, True, False, True, False])
    np.testing.assert_allclose(scores[:3], 0)
    assert scores[3] < -1
    assert scores[4] < scores[0]
    assert np.isneginf(scores[5])
    empty_scores, empty_front = episode_priorities(returns, jnp.zeros(6, jnp.bool_))
    assert np.isneginf(empty_scores).all()
    assert not empty_front.any()
    single_scores, single_front = episode_priorities(
        jnp.array([[2.0]]), jnp.array([True])
    )
    np.testing.assert_array_equal(single_front, [True])
    np.testing.assert_allclose(single_scores, [0])


def test_full_archive_retains_new_episode_and_removes_worst_old_episode():
    previous = archive([[4, 0], [0, 4], [-1, -1]])
    fresh = archive([[-3, -3]])
    actual = jax.jit(add_episodes)(previous, fresh)
    values = np.asarray(actual.returns[:, 0])
    np.testing.assert_array_equal(values, [[4, 0], [0, 4], [-3, -3]])
    # A new duplicate also survives insertion. At the next insertion, the
    # earlier duplicate representative wins over the duplicate old episode.
    duplicate = add_episodes(archive([[4, 0], [0, 4], [-1, -1]]), archive([[4, 0]]))
    updated = add_episodes(duplicate, archive([[2, 2]]))
    np.testing.assert_array_equal(updated.returns[:, 0], [[4, 0], [0, 4], [2, 2]])
    batch = add_episodes(previous, archive([[-3, -3], [-4, -4]]))
    # Fresh episodes retain positive insertion priority until the next ranking.
    assert any(np.array_equal(value, [-3, -3]) for value in batch.returns[:, 0])
    assert any(np.array_equal(value, [-4, -4]) for value in batch.returns[:, 0])


def test_archive_fills_empty_slots_before_evicting_and_samples_no_padding():
    previous = archive([[2, 3], [0, 0], [0, 0]], [1, 0, 0])
    previous = add_episodes(previous, archive([[4, 5]], [2]))
    np.testing.assert_array_equal(previous.lengths, [1, 2, 0])
    batch = jax.jit(lambda key: sample_episode_batch(previous, key, 128))(
        jax.random.key(0)
    )
    assert np.isfinite(batch.obs).all()
    episode = np.asarray(batch.actions)
    # Actions in the fresh fixture are zero; determine identity from return.
    second = np.asarray(batch.extras.desired_return[:, 0]) == 4
    times = np.asarray(batch.obs[:, 0]) % 10
    expected = np.where(second, 2, 1) - times
    np.testing.assert_array_equal(batch.extras.desired_horizon, expected)
    assert (times[~second] == 0).all()
    assert len(episode) == 128


def test_optimistic_commands_use_front_and_modify_only_one_coordinate():
    previous = archive([[0, 6], [6, 0], [-10, -10]], [1, 3, 2])
    desired, horizon = jax.jit(lambda key: sample_commands(previous, key, 128, 3))(
        jax.random.key(1)
    )
    desired = np.asarray(desired)
    for value in desired:
        differences = value - np.array([[0, 6], [6, 0]])
        assert any(
            (diff >= 0).all() and (diff <= 3).all() and (np.count_nonzero(diff) <= 1)
            for diff in differences
        )
    assert np.all(horizon == 1)
    one = archive([[2], [1]], [3, 1])
    desired, horizon = sample_commands(one, jax.random.key(2), 4, 2)
    assert desired.shape == (4, 1)
    np.testing.assert_array_equal(desired, 2)
    np.testing.assert_array_equal(horizon, 1)


@pytest.mark.parametrize("continuous", [False, True])
def test_supervised_loss_and_gradient_match_independent_values(continuous, jit):
    agent = PCNAgent(
        num_objectives=2,
        agent_network_cfg=OmegaConf.create(dict(hidden_size=3)),
        scaling_factor=(0.1, 0.2, 0.3),
    )
    action_space = (
        Box(low=-jnp.ones(2), high=jnp.ones(2)) if continuous else Discrete(n=2)
    )
    state = agent.init(
        Box(low=-jnp.ones(2), high=jnp.ones(2)), action_space, jax.random.key(0)
    )
    models = agent.get_models(state, copy=True)
    nnx.update(models, jax.tree.map(jnp.zeros_like, nnx.state(models, nnx.Param)))
    models.policy.output.bias[...] = jnp.array([2.0, -1.0])
    state = state.replace(model_state=nnx.unpack(models))
    actions = jnp.array([[0.5, -0.5], [-0.5, 0.5]]) if continuous else jnp.array([0, 1])
    batch = SampleBatch(
        obs=jnp.ones((2, 2)),
        actions=actions,
        extras=PyTreeDict(
            desired_return=jnp.array([[3.0, -2.0], [2.0, -1.0]]),
            desired_horizon=jnp.array([2.0, 1.0]),
        ),
    )
    loss_fn = lambda state: agent.loss(state, batch, jax.random.key(1)).policy_loss
    actual = (jax.jit(loss_fn) if jit else loss_fn)(state)
    logits = np.array([2.0, -1.0])
    if continuous:
        expected = np.square(logits - actions).mean()
        expected_grad = np.mean(logits - actions, axis=0)
    else:
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        expected = -np.log(probabilities).mean()
        expected_grad = probabilities - 0.5
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    grads = nnx.grad(
        lambda models: loss_fn(state.replace(model_state=nnx.unpack(models)))
    )(models)
    np.testing.assert_allclose(grads.policy.output.bias, expected_grad, rtol=2e-6)
    action, _ = agent.evaluate_actions(state, batch, jax.random.key(1))
    if continuous:
        np.testing.assert_array_equal(action, [[1, -1], [1, -1]])
    else:
        np.testing.assert_array_equal(action, [0, 0])


@pytest.mark.parametrize("continuous", [False, True])
def test_observation_normalization_matches_explicit_inputs_and_gradients(
    continuous, jit
):
    """Sampling, evaluation and supervised updates normalize only observations."""
    agent = PCNAgent(
        num_objectives=2,
        agent_network_cfg=OmegaConf.create(dict(hidden_size=4)),
        scaling_factor=(0.1, 0.2, 0.3),
        obs_preprocessor=running_statistics.normalize,
    )
    space = Box(low=-10 * jnp.ones(2), high=10 * jnp.ones(2))
    actions = space if continuous else Discrete(n=2)
    state = agent.init(space, actions, jax.random.key(11))
    stats = running_statistics.update(
        state.obs_preprocessor_state,
        jnp.array([[1.0, -4.0], [3.0, -2.0], [5.0, 0.0]]),
    )
    state = state.replace(obs_preprocessor_state=stats)
    batch = SampleBatch(
        obs=jnp.array([[4, -3], [2, -1]], jnp.int32),
        actions=jnp.array([[0.5, -0.5], [-0.5, 0.5]])
        if continuous
        else jnp.array([0, 1]),
        extras=PyTreeDict(
            desired_return=jnp.array([[3.0, -2.0], [2.0, -1.0]]),
            desired_horizon=jnp.array([2.0, 1.0]),
        ),
    )
    normalized = batch.replace(
        obs=running_statistics.normalize(batch.obs.astype(jnp.float32), stats)
    )
    reference_agent = agent.replace(obs_preprocessor=None)

    def forward(state, batch):
        return (
            agent.predictions(state, batch),
            agent.loss(state, batch, jax.random.key(12)),
            agent.compute_actions(state, batch, jax.random.key(12)),
            agent.evaluate_actions(state, batch, jax.random.key(12)),
        )

    actual = (jax.jit(forward) if jit else forward)(state, batch)
    expected = (
        reference_agent.predictions(state, normalized),
        reference_agent.loss(state, normalized, jax.random.key(12)),
        reference_agent.compute_actions(state, normalized, jax.random.key(12)),
        reference_agent.evaluate_actions(state, normalized, jax.random.key(12)),
    )
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    models = agent.get_models(state, copy=True)
    actual_grad = nnx.grad(
        lambda models: (
            agent.loss(
                state.replace(model_state=nnx.unpack(models)), batch, jax.random.key(12)
            ).policy_loss
        )
    )(models)
    expected_grad = nnx.grad(
        lambda models: (
            reference_agent.loss(
                state.replace(model_state=nnx.unpack(models)),
                normalized,
                jax.random.key(12),
            ).policy_loss
        )
    )(models)
    for a, b in zip(
        jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
