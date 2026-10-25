"""MORL workflows preserve vector rewards, preference state and functional updates."""

from pathlib import Path
from unittest.mock import Mock, patch

from hydra import compose, initialize_config_dir
from hydra.utils import get_class
import jax
import jax.numpy as jnp
import chex
from flax import nnx
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.envs import AutoresetMode
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.orbax_utils import load, save
from evorl.envs.wrappers import (
    ActionSquashWrapper,
    EpisodeWrapper,
    OneEpisodeWrapper,
    VmapAutoResetWrapper,
    VmapWrapper,
)
from tests.helpers.multi_objective_envs import ToyMultiObjectiveEnv

pytestmark = pytest.mark.integration


def make_workflow(
    algorithm, jit=True, multi_device=False, objectives=2, overrides=None
):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config", overrides=[f"agent=multi_objective/{algorithm}"]
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            num_envs=4,
            num_eval_envs=4,
            eval_episodes=4,
            rollout_length=2,
            random_timesteps=8,
            learning_start_timesteps=8,
            batch_size=4,
            replay_buffer_capacity=64,
            num_updates_per_iter=2,
            normalize_obs=True,
            total_timesteps=24,
            fold_iters=1,
            eval_interval=100,
            agent_network=dict(
                actor_hidden_layer_sizes=[4],
                critic_hidden_layer_sizes=[4],
                q_hidden_layer_sizes=[4],
            ),
        ),
        overrides or {},
    )
    base = get_class(config.workflow_cls)
    cls = type(f"Test{algorithm}Workflow", (base,), {})

    def create_env(config, parallel, autoreset_mode, **kwargs):
        native = ToyMultiObjectiveEnv(algorithm == "capql", objectives)
        if algorithm == "capql":
            native = ActionSquashWrapper(native)
        if autoreset_mode == AutoresetMode.DISABLED:
            return VmapWrapper(
                OneEpisodeWrapper(native, config.max_episode_steps),
                parallel,
                vmap_step=True,
            )
        return VmapAutoResetWrapper(
            EpisodeWrapper(native, config.max_episode_steps), parallel
        )

    with patch(
        f"evorl.algorithms.multi_objective.{algorithm}.create_env",
        side_effect=create_env,
    ):
        return cls.build_from_config(config, jit=jit, multi_device=multi_device)


@pytest.mark.parametrize("algorithm", ["envelope_q", "capql"])
@pytest.mark.parametrize("objectives", [1, 2, 3])
def test_vector_workflow_training_evaluation_and_input_isolation(
    algorithm, objectives, jit
):
    workflow = make_workflow(algorithm, jit=jit, objectives=objectives)
    try:
        state = workflow.setup(jax.random.key(0))
        assert state.replay_buffer_state.data.next_obs is None
        assert state.replay_buffer_state.data.dones is None
        before = jax.tree.map(np.array, (state.agent_state, state.opt_state))
        metrics, updated = workflow.step(state)
        assert np.isfinite(metrics.loss)
        assert int(updated.metrics.sampled_timesteps) == 16
        assert int(state.metrics.sampled_episodes) == 0
        assert int(updated.metrics.sampled_episodes) == 4
        assert int(updated.metrics.training_updates) == 2
        assert updated.rollout_preferences.shape == (4, objectives)
        np.testing.assert_allclose(updated.rollout_preferences.sum(-1), 1, atol=1e-6)
        assert updated.replay_buffer_state.data.rewards.shape == (64, objectives)
        assert updated.replay_buffer_state.data.next_obs is None
        assert updated.replay_buffer_state.data.dones is None
        assert updated.replay_buffer_state.data.extras.env_extras.ori_obs.shape == (
            64,
            3,
        )
        assert updated.replay_buffer_state.data.extras.env_extras.termination.shape == (
            64,
        )
        assert updated.replay_buffer_state.data.extras.preferences.shape == (
            64,
            objectives,
        )
        assert any(
            not np.array_equal(x, y)
            for x, y in zip(
                jax.tree.leaves(state.agent_state.model_state),
                jax.tree.leaves(updated.agent_state.model_state),
                strict=True,
            )
        )
        evaluation, _ = workflow.evaluate(updated)
        num_preferences = 1 if objectives == 1 else objectives + 1
        assert evaluation.episode_returns.shape == (num_preferences, objectives)
        assert evaluation.episode_lengths.shape == (num_preferences,)
        np.testing.assert_allclose(evaluation.episode_lengths, 2)
        np.testing.assert_allclose(
            evaluation.scalarized_returns,
            (evaluation.preferences * evaluation.episode_returns).sum(-1),
        )
        for previous, current in zip(
            jax.tree.leaves(before),
            jax.tree.leaves((state.agent_state, state.opt_state)),
            strict=True,
        ):
            np.testing.assert_array_equal(previous, current)
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["envelope_q", "capql"])
def test_preferences_change_at_the_algorithm_boundaries(algorithm):
    workflow = make_workflow(algorithm)
    try:
        state = workflow.setup(jax.random.key(6))
        trajectory, state = jax.jit(
            lambda state: workflow._rollout(state, jax.random.key(7), 3)
        )(state)
        preferences = trajectory.extras.preferences
        # Terminal bootstrap must see the old episode's final observation,
        # while the following policy call sees the reset observation.
        np.testing.assert_array_equal(trajectory.extras.env_extras.ori_obs[1, :, 0], 2)
        np.testing.assert_array_equal(trajectory.obs[2, :, 0], 0)
        np.testing.assert_array_equal(trajectory.extras.env_extras.termination[1], 1)
        if algorithm == "envelope_q":
            np.testing.assert_array_equal(preferences[0], preferences[1])
        else:
            assert not np.array_equal(preferences[0], preferences[1])
        assert not np.array_equal(preferences[1], preferences[2])
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["envelope_q", "capql"])
def test_morl_two_device_training(algorithm, mesh_factory):
    mesh_factory()
    workflow = make_workflow(algorithm, multi_device=True)
    try:
        with jax.set_mesh(workflow.mesh):
            state = workflow.setup(jax.random.key(3))
            metrics, state = workflow.step(state)
            evaluation, _ = workflow.evaluate(state)
        assert np.isfinite(metrics.loss)
        assert int(state.metrics.sampled_timesteps) == 16
        assert state.rollout_preferences.shape == (4, 2)
        assert not state.rollout_preferences.sharding.is_fully_replicated
        for value in jax.tree.leaves(state.agent_state):
            assert value.sharding.is_fully_replicated
        assert evaluation.episode_returns.shape == (3, 2)
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["envelope_q", "capql"])
def test_morl_nested_populations_match_independent_training(algorithm):
    workflow = make_workflow(algorithm, jit=False)
    try:

        def train(key):
            state = workflow.setup(key)
            return workflow.step(state)

        keys = jax.random.split(jax.random.key(4), (2, 2))
        metrics, state = jax.jit(jax.vmap(jax.vmap(train)))(keys)
        reference_metrics, reference = jax.jit(train)(keys[0, 0])
        np.testing.assert_allclose(
            metrics.loss[0, 0], reference_metrics.loss, atol=2e-5, rtol=2e-5
        )
        for actual, expected in zip(
            jax.tree.leaves(state.agent_state),
            jax.tree.leaves(reference.agent_state),
            strict=True,
        ):
            np.testing.assert_allclose(actual[0, 0], expected, atol=2e-5, rtol=2e-5)
        evaluation, _ = jax.jit(jax.vmap(jax.vmap(workflow.evaluate)))(state)
        assert evaluation.episode_returns.shape == (2, 2, 3, 2)
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["envelope_q", "capql"])
def test_morl_learn_budget_and_logging(algorithm):
    workflow = make_workflow(algorithm)
    workflow.recorder, workflow.checkpoint_manager = Mock(), Mock()
    try:
        state = workflow.init(jax.random.key(9))
        workflow.recorder.write.assert_not_called()
        state = workflow.learn(state)
        assert int(state.metrics.sampled_timesteps) == 24
        assert int(state.metrics.iterations) == 2
        data = workflow.recorder.write.call_args.args[0]
        assert data["eval/episode_returns"].shape == (3, 2)
        assert data["eval/preferences"].shape == (3, 2)
        assert workflow.checkpoint_manager.save.call_args.kwargs["force"] is True
        workflow.recorder.reset_mock()
        state = workflow.learn(state)
        workflow.recorder.write.assert_not_called()
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["envelope_q", "capql"])
def test_morl_checkpoint_replays_models_preferences_and_local_ring(tmp_path, algorithm):
    workflow = make_workflow(algorithm)
    try:
        state = workflow.setup(jax.random.key(31))
        _, state = workflow.step(state)
        save(tmp_path / "checkpoint", state)
        restored = load(tmp_path / "checkpoint", state)
        expected_metrics, expected = workflow.step(state)
        actual_metrics, actual = workflow.step(restored)
        chex.assert_trees_all_equal(actual_metrics, expected_metrics)
        chex.assert_trees_all_equal(actual, expected)
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["envelope_q", "capql"])
@pytest.mark.parametrize("random_steps,policy_start", [(0, 8), (8, 4), (4, 8)])
def test_morl_warmup_handles_empty_phases(algorithm, random_steps, policy_start):
    workflow = make_workflow(
        algorithm,
        overrides=dict(
            random_timesteps=random_steps, learning_start_timesteps=policy_start
        ),
    )
    try:
        state = workflow.setup(jax.random.key(35))
        assert int(state.metrics.sampled_timesteps) == 8
        assert int(state.replay_buffer_state.buffer_size) == 8
        np.testing.assert_array_equal(state.env_state.env_state, 0)
        metrics, state = workflow.step(state)
        assert np.isfinite(metrics.loss)
        assert int(state.metrics.sampled_timesteps) == 16
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["envelope_q", "capql"])
def test_morl_long_warmup_passes_the_full_batch_to_replay(algorithm, monkeypatch):
    original_step = ToyMultiObjectiveEnv.step

    def nonterminating_step(self, state, action):
        return original_step(self, state, action).replace(done=jnp.float32(0))

    monkeypatch.setattr(ToyMultiObjectiveEnv, "step", nonterminating_step)
    workflow = make_workflow(
        algorithm,
        overrides=dict(
            env=dict(max_episode_steps=1000),
            random_timesteps=20,
            learning_start_timesteps=20,
            replay_buffer_capacity=8,
        ),
    )
    try:
        buffer_cls = type(workflow.replay_buffer)
        original_add = buffer_cls.add
        batch_sizes = []

        def add(self, buffer, trajectory):
            batch_sizes.append(trajectory.obs.shape[0])
            return original_add(self, buffer, trajectory)

        monkeypatch.setattr(buffer_cls, "add", add)
        state = workflow.setup(jax.random.key(36))
        assert batch_sizes == [20]
        assert int(state.metrics.sampled_timesteps) == 20
        assert int(state.replay_buffer_state.buffer_size) == 8
        assert int(state.replay_buffer_state.current_index) == 4
        np.testing.assert_array_equal(state.env_state.env_state, 0)
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["envelope_q", "capql"])
def test_morl_target_updates_and_envelope_schedules_count_gradient_steps(algorithm):
    workflow = make_workflow(
        algorithm,
        overrides=dict(
            num_updates_per_iter=1,
            target_network_update_freq=3,
            tau=0.2,
            exploration_epsilon=dict(start=1.0, end=0.0, decay_updates=3),
            homotopy_lambda=dict(start=0.0, end=1.0, decay_updates=3),
        ),
    )
    try:
        state = workflow.setup(jax.random.key(37))
        target_name, online_name = (
            ("target_q", "q")
            if algorithm == "envelope_q"
            else ("target_critic", "critic")
        )
        before = workflow.agent.get_models(state.agent_state, copy=True)
        original = nnx.state(getattr(before, target_name), nnx.Param)
        for update in range(1, 4):
            _, state = workflow.step(state)
            models = workflow.agent.get_models(state.agent_state, copy=True)
            target = nnx.state(getattr(models, target_name), nnx.Param)
            online = nnx.state(getattr(models, online_name), nnx.Param)
            for old, actual, source in zip(
                jax.tree.leaves(original),
                jax.tree.leaves(target),
                jax.tree.leaves(online),
                strict=True,
            ):
                expected = 0.8 * old + 0.2 * source if update == 3 else old
                np.testing.assert_allclose(actual, expected, atol=1e-7)
            if algorithm == "envelope_q":
                np.testing.assert_allclose(
                    state.agent_state.extra_state.exploration_epsilon,
                    1 - update / 3,
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    state.agent_state.extra_state.homotopy_lambda, update / 3, atol=1e-7
                )
    finally:
        workflow.close()


@pytest.mark.parametrize(
    "example",
    ["envelope_q/deep-sea-treasure", "capql/brax-ant", "capql/playground-cartpole"],
)
def test_morl_yaml_runs_training_and_evaluation_on_real_environments(example):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config", overrides=[f"agent=exp/multi_objective/{example}"]
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            num_envs=2,
            num_eval_envs=2,
            eval_episodes=2,
            rollout_length=1,
            random_timesteps=2,
            learning_start_timesteps=2,
            batch_size=2,
            replay_buffer_capacity=16,
            num_updates_per_iter=2,
            normalize_obs=True,
            agent_network=dict(
                actor_hidden_layer_sizes=[4],
                critic_hidden_layer_sizes=[4],
                q_hidden_layer_sizes=[4],
            ),
        ),
    )
    base = get_class(config.workflow_cls)
    cls = type(f"Real{example}Workflow", (base,), {})
    workflow = cls.build_from_config(config, jit=True)
    try:
        state = workflow.setup(jax.random.key(12))
        metrics, state = workflow.step(state)
        evaluation, _ = workflow.evaluate(state)
        assert np.isfinite(metrics.loss)
        assert int(state.metrics.sampled_timesteps) == 4
        assert evaluation.episode_returns.shape == (5, 2)
        assert np.isfinite(evaluation.episode_returns).all()
        assert np.all(evaluation.episode_lengths > 0)
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["envelope_q", "capql"])
def test_morl_learns_opposing_preferences_in_a_quadratic_or_discrete_bandit(algorithm):
    workflow = make_workflow(
        algorithm,
        overrides=dict(
            discount=0.0,
            alpha=0.05,
            normalize_obs=False,
            num_envs=16,
            num_eval_envs=16,
            eval_episodes=16,
            rollout_length=1,
            random_timesteps=64,
            learning_start_timesteps=64,
            batch_size=32,
            replay_buffer_capacity=1024,
            num_updates_per_iter=2,
            total_timesteps=10000,
            optimizer=dict(lr=0.003),
            agent_network=dict(
                actor_hidden_layer_sizes=[16],
                critic_hidden_layer_sizes=[16],
                q_hidden_layer_sizes=[16],
            ),
        ),
    )
    try:
        state = workflow.setup(jax.random.key(23))

        def train(state, _):
            _, state = workflow.step(state)
            return state, None

        state, _ = jax.jit(lambda state: jax.lax.scan(train, state, (), length=250))(
            state
        )
        batch = SampleBatch(
            obs=jnp.tile(jnp.array([0.0, 1.0, -1.0]), (2, 1)),
            extras=PyTreeDict(preferences=jnp.eye(2)),
        )
        actions, _ = workflow.agent.evaluate_actions(
            state.agent_state, batch, jax.random.key(24)
        )
        if algorithm == "envelope_q":
            np.testing.assert_array_equal(actions, [0, 1])
        else:
            assert float(actions[0, 0]) > 0.4
            assert float(actions[1, 0]) < -0.4
    finally:
        workflow.close()
