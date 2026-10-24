"""PGMORL prediction, task selection, archive and resumable training."""

from pathlib import Path
from unittest.mock import Mock, patch

from hydra import compose, initialize_config_dir
import jax
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.algorithms.multi_objective.pgmorl import (
    PGMORLWorkflow,
    generate_weights,
    hypervolume,
    pareto_indices,
    predict_performance,
    sparsity,
)
from evorl.envs import AutoresetMode
from evorl.envs.wrappers import (
    ActionSquashWrapper,
    EpisodeWrapper,
    OneEpisodeWrapper,
    VmapEnvPoolAutoResetWrapper,
    VmapWrapper,
)
from evorl.utils.orbax_utils import load, save
from tests.helpers.multi_objective_envs import ToyMultiObjectiveEnv


def test_simplex_and_maximization_front_geometry():
    weights = generate_weights(2, 3)
    assert weights.shape == (6, 3)
    np.testing.assert_array_equal(weights.sum(-1), 1)
    values = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    np.testing.assert_array_equal(pareto_indices(values), [0, 1, 2])
    np.testing.assert_allclose(hypervolume(values, [0.0, 0.0]), 6.0)
    np.testing.assert_allclose(hypervolume([[1, 2, 3], [2, 1, 3]], [0, 0, 0]), 9.0)
    np.testing.assert_allclose(sparsity(values[:3]), 2.0)


def test_hyperbolic_predictor_handles_zero_coordinates_and_identical_histories():
    weights = np.stack((np.linspace(0, 1, 16), np.linspace(1, 0, 16)), -1)
    before = np.zeros_like(weights)
    delta = np.stack(
        (
            2 * np.tanh(2 * (weights[:, 0] - 0.5)) + 0.2,
            3 * np.tanh(2 * (weights[:, 1] - 0.5)) + 0.3,
        ),
        -1,
    )
    predicted = predict_performance(
        weights, before, before + delta, np.zeros(2), np.array([0.75, 0.25]), sigma=1.0
    )
    expected = [2 * np.tanh(0.5) + 0.2, 3 * np.tanh(-0.5) + 0.3]
    np.testing.assert_allclose(predicted, expected, atol=0.03)
    identical = predict_performance(
        weights[:4], before[:4], before[:4], np.zeros(2), [0.5, 0.5]
    )
    assert np.isfinite(identical).all()
    np.testing.assert_allclose(identical, 0, atol=1e-5)
    np.testing.assert_allclose(
        predict_performance(
            weights[:1], before[:1], np.ones((1, 2)), np.zeros(2), [0.5, 0.5]
        ),
        [1, 1],
    )


def make_pgmorl(total_timesteps=40, jit=True, objectives=2):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config", overrides=["agent=multi_objective/pgmorl"]
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            num_envs=2,
            num_eval_envs=2,
            eval_episodes=2,
            rollout_length=2,
            minibatch_size=4,
            reuse_rollout_epochs=1,
            total_timesteps=total_timesteps,
            pop_size=2,
            warmup_iterations=1,
            evolutionary_iterations=1,
            performance_buffer_divisions=2,
            performance_buffer_size=1,
            weight_divisions=2,
            archive_capacity=8,
            reference_point=[-20.0] * objectives,
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
    )

    def create_env(config, parallel, autoreset_mode, **kwargs):
        env = ActionSquashWrapper(ToyMultiObjectiveEnv(True, objectives))
        if autoreset_mode == AutoresetMode.DISABLED:
            return VmapWrapper(
                OneEpisodeWrapper(env, config.max_episode_steps),
                parallel,
                vmap_step=True,
            )
        assert autoreset_mode == AutoresetMode.ENVPOOL
        return VmapEnvPoolAutoResetWrapper(
            EpisodeWrapper(env, config.max_episode_steps), parallel
        )

    with patch(
        "evorl.algorithms.multi_objective.moppo.create_env", side_effect=create_env
    ):
        return PGMORLWorkflow.build_from_config(config, jit=jit)


@pytest.mark.integration
def test_prediction_guided_training_preserves_history_archive_and_budget(tmp_path):
    workflow = make_pgmorl(20)
    workflow.recorder = Mock()
    workflow.checkpoint_manager = Mock()
    try:
        state = workflow.setup(jax.random.key(0))
        initial = state
        assert int(state.archive_count) >= 1
        assert bool(state.pool_valid.any())
        metric, state = workflow.step(state)
        assert np.isfinite(metric.loss)
        assert int(state.metrics.sampled_timesteps) == 8
        assert int(state.history_count) == 2
        assert int(initial.history_count) == 0
        np.testing.assert_array_equal(state.pool_valid.shape, initial.pool_valid.shape)
        assert np.isfinite(np.asarray(state.history_after)).all()
        path = tmp_path / "pgmorl"
        save(path, state)
        restored = load(path, state)
        _, expected = workflow.step(state)
        _, actual = workflow.step(restored)
        for x, y in zip(
            jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
        ):
            if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
                x, y = jax.random.key_data(x), jax.random.key_data(y)
            np.testing.assert_allclose(x, y)
        state = workflow.learn(actual)
        assert int(state.metrics.sampled_timesteps) == 20
        assert int(state.metrics.training_updates) == 5
        assert workflow.recorder.write.called
        assert workflow.checkpoint_manager.save.call_args.kwargs["force"]
        assert int(state.history_count) == 5
        archive = np.asarray(state.archive_returns)[: int(state.archive_count)]
        assert len(pareto_indices(archive)) == len(archive)
        evaluation, _ = workflow.evaluate(state)
        assert evaluation.episode_returns.shape[1] == 2
        np.testing.assert_array_equal(evaluation.episode_lengths, 2)
        workflow.recorder.reset_mock()
        workflow.learn(state)
        workflow.recorder.write.assert_not_called()
    finally:
        workflow.close()


@pytest.mark.integration
def test_pgmorl_task_assignment_keeps_destination_environment_and_keys():
    workflow = make_pgmorl()
    try:
        state = workflow.setup(jax.random.key(4))
        _, state = workflow.step(state)
        selected = workflow._select_tasks(state)
        for x, y in zip(
            jax.tree.leaves(state.population.env_state),
            jax.tree.leaves(selected.population.env_state),
        ):
            if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
                x, y = jax.random.key_data(x), jax.random.key_data(y)
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(
            jax.random.key_data(state.population.key),
            jax.random.key_data(selected.population.key),
        )
        np.testing.assert_allclose(
            selected.population.agent_state.extra_state.preferences.sum(-1), 1
        )
    finally:
        workflow.close()


@pytest.mark.integration
def test_pgmorl_eager_execution_does_not_compile_population_helpers():
    workflow = make_pgmorl(8, jit=False)
    try:
        assert not hasattr(workflow.inner.step, "lower")
        with patch(
            "evorl.algorithms.multi_objective.pgmorl.jax.jit", wraps=jax.jit
        ) as compile_fn:
            state = workflow.setup(jax.random.key(15))
            metrics, state = workflow.step(state)
            evaluation, _ = workflow.evaluate(state)
        assert np.isfinite(metrics.loss)
        assert np.isfinite(evaluation.episode_returns).all()
        compile_fn.assert_not_called()
    finally:
        workflow.close()


@pytest.mark.integration
def test_pgmorl_three_objective_training_and_archive():
    workflow = make_pgmorl(8, objectives=3)
    try:
        state = workflow.setup(jax.random.key(16))
        metrics, state = workflow.step(state)
        evaluation, _ = workflow.evaluate(state)
        assert np.isfinite(metrics.loss)
        assert state.performances.shape == (2, 3)
        assert evaluation.episode_returns.shape[-1] == 3
        assert np.isfinite(evaluation.episode_returns).all()
        np.testing.assert_allclose(
            state.population.agent_state.extra_state.preferences.sum(-1), 1
        )
    finally:
        workflow.close()
