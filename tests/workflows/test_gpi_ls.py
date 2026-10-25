"""GPI-LS functional training and host linear-support stage integration."""

from pathlib import Path
from unittest.mock import Mock, patch

import chex
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.algorithms.multi_objective.gpi_ls import GPILSWorkflow
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.envs import AutoresetMode
from evorl.envs.wrappers import (
    EpisodeWrapper,
    OneEpisodeWrapper,
    VmapAutoResetWrapper,
    VmapWrapper,
)
from evorl.utils.orbax_utils import load, save
from tests.helpers.multi_objective_envs import ToyMultiObjectiveEnv

pytestmark = pytest.mark.integration


def make_workflow(jit=True, multi_device=False, real=False, **overrides):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config",
            overrides=["agent=exp/multi_objective/gpi_ls/deep-sea-treasure"],
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            num_envs=4,
            num_eval_envs=4,
            eval_episodes=4,
            rollout_length=2,
            random_timesteps=8,
            learning_start_timesteps=8,
            replay_buffer_capacity=64,
            batch_size=4,
            num_updates_per_iter=2,
            fold_iters=1,
            total_timesteps=24,
            eval_interval=100,
            max_support_size=8,
            training_steps_per_weight=1,
            agent_network=dict(q_hidden_layer_sizes=[4]),
            env=dict(max_episode_steps=3),
        ),
        overrides,
    )
    cls = type("TestGPILSWorkflow", (GPILSWorkflow,), {})
    if real:
        return cls.build_from_config(config, jit=jit, multi_device=multi_device)

    def create_env(config, parallel, autoreset_mode, **kwargs):
        native = ToyMultiObjectiveEnv()
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
        "evorl.algorithms.multi_objective.gpi_ls.create_env", side_effect=create_env
    ):
        return cls.build_from_config(config, jit=jit, multi_device=multi_device)


@pytest.mark.parametrize("compiled", [False, True])
def test_training_support_selection_evaluation_and_state_isolation(compiled):
    workflow = make_workflow(jit=compiled)
    try:
        state = workflow.setup(jax.random.key(0))
        assert state.replay_buffer_state.data.next_obs is None
        assert state.replay_buffer_state.data.dones is None
        before = jax.tree.map(np.array, (state.agent_state, state.opt_state))
        metrics, trained = workflow.step(state)
        assert np.isfinite(metrics.loss)
        assert int(trained.metrics.sampled_timesteps) == 16
        assert int(trained.metrics.training_updates) == 2
        assert trained.replay_buffer_state.data.next_obs is None
        assert trained.replay_buffer_state.data.dones is None
        updated = workflow.update_linear_support(trained)
        assert int(updated.agent_state.extra_state.support_iterations) == 1
        np.testing.assert_array_equal(
            updated.agent_state.extra_state.training_preference, [0, 1]
        )
        assert updated.agent_state.extra_state.coverage_mask.any()
        evaluation, _ = workflow.evaluate(updated)
        assert evaluation.episode_returns.shape == (5, 2)
        for old, current in zip(
            jax.tree.leaves(before),
            jax.tree.leaves((state.agent_state, state.opt_state)),
            strict=True,
        ):
            np.testing.assert_array_equal(old, current)
    finally:
        workflow.close()


def test_prefill_keeps_data_but_starts_training_from_a_new_episode(jit):
    workflow = make_workflow(jit=jit, random_timesteps=4, learning_start_timesteps=12)
    try:
        state = workflow.setup(jax.random.key(20))
        np.testing.assert_array_equal(state.env_state.env_state, 0)
        assert int(state.metrics.sampled_timesteps) == 12
        assert int(state.replay_buffer_state.buffer_size) == 12
        assert int(state.metrics.sampled_episodes) == 0
        _, state = workflow.step(state)
        assert int(state.metrics.sampled_timesteps) == 20
        assert int(state.metrics.sampled_episodes) == 4
    finally:
        workflow.close()


def test_learn_runs_linear_support_stages_budget_and_records():
    workflow = make_workflow()
    workflow.recorder, workflow.checkpoint_manager = Mock(), Mock()
    try:
        state = workflow.init(jax.random.key(1))
        workflow.recorder.write.assert_not_called()
        state = workflow.learn(state)
        assert int(state.metrics.sampled_timesteps) == 24
        assert int(state.metrics.iterations) == 2
        assert int(state.agent_state.extra_state.support_iterations) == 2
        assert workflow.checkpoint_manager.save.call_args.kwargs["force"] is True
        assert workflow.recorder.write.call_args.args[0][
            "eval/episode_returns"
        ].shape == (5, 2)
        workflow.recorder.reset_mock()
        workflow.learn(state)
        workflow.recorder.write.assert_not_called()
    finally:
        workflow.close()


def test_checkpoint_restores_support_scheduler_and_next_update(tmp_path):
    workflow = make_workflow()
    try:
        state = workflow.setup(jax.random.key(3))
        _, state = workflow.step(state)
        state = workflow.update_linear_support(state)
        save(tmp_path / "checkpoint", state)
        restored = load(tmp_path / "checkpoint", state)
        chex.assert_trees_all_equal(workflow.step(state), workflow.step(restored))
        _, next_state = workflow.step(state)
        _, next_restored = workflow.step(restored)
        chex.assert_trees_all_equal(
            workflow.update_linear_support(next_state),
            workflow.update_linear_support(next_restored),
        )
    finally:
        workflow.close()


def test_jitted_nested_populations_match_independent_training():
    workflow = make_workflow(jit=False)
    try:

        def train(key):
            state = workflow.setup(key)
            return workflow.step(state)

        keys = jax.random.split(jax.random.key(4), (2, 2))
        metric, state = jax.jit(jax.vmap(jax.vmap(train)))(keys)
        reference_metric, reference = jax.jit(train)(keys[0, 0])
        np.testing.assert_allclose(metric.loss[0, 0], reference_metric.loss, rtol=2e-5)
        for actual, expected in zip(
            jax.tree.leaves(state.agent_state),
            jax.tree.leaves(reference.agent_state),
            strict=True,
        ):
            np.testing.assert_allclose(actual[0, 0], expected, rtol=2e-5, atol=2e-5)
        evaluated, _ = jax.jit(jax.vmap(jax.vmap(workflow.evaluate)))(state)
        assert evaluated.episode_returns.shape == (2, 2, 5, 2)
    finally:
        workflow.close()


def test_two_device_training_and_host_linear_support(dp_mesh):
    workflow = make_workflow(multi_device=True)
    try:
        with jax.set_mesh(workflow.mesh):
            state = workflow.setup(jax.random.key(5))
            metrics, state = workflow.step(state)
            state = workflow.update_linear_support(state)
            evaluated, _ = workflow.evaluate(state)
        assert np.isfinite(metrics.loss)
        assert int(state.metrics.sampled_timesteps) == 16
        assert evaluated.episode_returns.shape == (5, 2)
        assert not state.rollout_preferences.sharding.is_fully_replicated
        for value in jax.tree.leaves(state.agent_state):
            assert value.sharding.is_fully_replicated
    finally:
        workflow.close()


def test_yaml_real_deep_sea_treasure_training_and_support():
    workflow = make_workflow(real=True)
    try:
        state = workflow.setup(jax.random.key(6))
        metrics, state = workflow.step(state)
        state = workflow.update_linear_support(state)
        evaluated, _ = workflow.evaluate(state)
        assert np.isfinite(metrics.loss)
        assert np.isfinite(evaluated.episode_returns).all()
        assert int(state.agent_state.extra_state.support_iterations) == 1
    finally:
        workflow.close()


def test_successor_features_learn_opposing_preferences_in_a_discrete_bandit():
    workflow = make_workflow(
        discount=0.0,
        num_envs=16,
        num_eval_envs=16,
        eval_episodes=16,
        rollout_length=1,
        random_timesteps=64,
        learning_start_timesteps=64,
        batch_size=32,
        replay_buffer_capacity=1024,
        num_updates_per_iter=2,
        optimizer=dict(lr=0.003),
        agent_network=dict(q_hidden_layer_sizes=[16]),
    )
    try:
        state = workflow.setup(jax.random.key(12))

        def train(state, _):
            _, state = workflow.step(state)
            return state, None

        state, _ = jax.jit(lambda state: jax.lax.scan(train, state, (), length=200))(
            state
        )
        actions, _ = workflow.agent.evaluate_actions(
            state.agent_state,
            SampleBatch(
                obs=jnp.tile(jnp.array([0.0, 1.0, -1.0]), (2, 1)),
                extras=PyTreeDict(preferences=jnp.eye(2)),
            ),
            jax.random.key(13),
        )
        np.testing.assert_array_equal(actions, [0, 1])
    finally:
        workflow.close()


@pytest.mark.parametrize("nested", [False, True])
def test_real_mo_gymnasium_jitted_training_and_evaluation(mo_gymnasium_pools, nested):
    workflow = make_workflow(
        jit=not nested,
        real=True,
        env=dict(
            env_type="mo_gymnasium",
            env_name="deep-sea-treasure-v0",
            max_episode_steps=3,
        ),
        num_envs=2,
        num_eval_envs=1,
        eval_episodes=1,
        batch_size=2,
        random_timesteps=4,
        learning_start_timesteps=4,
        eval_preferences=[[1.0, 0.0], [0.0, 1.0]],
    )
    try:

        def run(key):
            state = workflow.setup(key)
            train, state = workflow.step(state)
            evaluation, state = workflow.evaluate(state)
            return train, evaluation, state

        keys = jax.random.key(14)
        if nested:
            run = jax.jit(jax.vmap(jax.vmap(run)))
            keys = jax.random.split(keys, (2, 2))
        train, evaluation, state = run(keys)
        shape = (2, 2) if nested else ()
        assert np.isfinite(train.loss).all()
        assert evaluation.episode_returns.shape == (*shape, 2, 2)
        np.testing.assert_array_equal(
            state.metrics.sampled_timesteps, np.full(shape, 8)
        )
        if not nested:
            state = workflow.update_linear_support(state)
            assert state.agent_state.extra_state.coverage_mask.any()
    finally:
        workflow.close()


def test_stage_change_preserves_in_progress_episode_preference_and_final_observation():
    workflow = make_workflow(rollout_length=1)
    try:
        state = workflow.setup(jax.random.key(30))
        _, state = workflow.step(state)
        old_preferences = np.asarray(state.rollout_preferences)
        state = workflow.update_linear_support(state)
        np.testing.assert_array_equal(state.rollout_preferences, old_preferences)
        trajectory, _ = jax.jit(
            lambda state: workflow._rollout(state, jax.random.key(31), 2)
        )(state)
        np.testing.assert_array_equal(trajectory.extras.preferences[0], old_preferences)
        np.testing.assert_array_equal(trajectory.extras.env_extras.ori_obs[0, :, 0], 2)
        np.testing.assert_array_equal(trajectory.obs[1, :, 0], 0)
    finally:
        workflow.close()


def test_linear_support_selects_the_corner_with_largest_gpi_improvement():
    workflow = make_workflow()
    try:
        state = workflow.setup(jax.random.key(32))
        extra = state.agent_state.extra_state
        weights = np.zeros((8, 2), np.float32)
        weights[:2] = np.eye(2)
        returns = np.zeros_like(weights)
        returns[:2] = [[3.0, 0.0], [0.0, 3.0]]
        state = state.replace(
            agent_state=state.agent_state.replace(
                extra_state=extra.replace(
                    support_weights=jnp.asarray(weights),
                    support_mask=jnp.arange(8) < 2,
                    coverage_weights=jnp.asarray(weights),
                    coverage_returns=jnp.asarray(returns),
                    coverage_mask=jnp.arange(8) < 2,
                    support_iterations=jnp.uint32(2),
                )
            )
        )

        def values(agent_state, key):
            weight = np.asarray(agent_state.extra_state.preferences)
            if np.allclose(weight, [1, 0]):
                return jnp.array([3.0, 0.0])
            if np.allclose(weight, [0, 1]):
                return jnp.array([0.0, 3.0])
            if weight[0] > 0.5:
                return jnp.array([2.5, 1.5])
            return jnp.array([2.0, 2.0])

        with patch.object(workflow, "_evaluate_support", side_effect=values):
            updated = workflow.update_linear_support(state)
        np.testing.assert_allclose(
            updated.agent_state.extra_state.training_preference,
            [2 / 3, 1 / 3],
            atol=1e-6,
        )
        coverage = updated.agent_state.extra_state.coverage_returns[
            updated.agent_state.extra_state.coverage_mask
        ]
        assert any(np.allclose(value, [2, 2]) for value in coverage)
    finally:
        workflow.close()
