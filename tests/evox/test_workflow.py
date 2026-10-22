"""EvoX lifecycle, compilation hooks, accounting and checkpoint contracts."""

from unittest.mock import Mock, patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf

torch = pytest.importorskip("torch", minversion="2.9.0")
pytest.importorskip("evox")

from evox.core import Algorithm, Mutable

from evorl.algorithms.contrib.evox.evox_algorithm_wrapper import EvoXAlgorithmWrapper
from evorl.algorithms.contrib.evox.evox_workflow import EvoXWorkflow
from evorl.algorithms.ec.obs_utils import init_obs_preprocessor
from evorl.utils.orbax_utils import load, save
from evorl.workflows import Workflow
from tests.helpers.envs import ToyEnv, VectorRewardEnv
from tests.helpers.evox import (
    NSGA2TestWorkflow,
    workflow_config,
    workflow_state_with_key_data,
)


class MultipleEvaluationAlgorithm(Algorithm):
    def __init__(self, pop_size, lb, ub, device):
        super().__init__()
        self.pop = Mutable(
            torch.rand(pop_size, lb.numel(), device=device) * (ub - lb) + lb
        )
        self.fit = Mutable(torch.zeros(pop_size, device=device))

    def step(self):
        self.evaluate(self.pop[:2])
        self.fit = self.evaluate(self.pop)


class MultipleEvaluationWorkflow(EvoXWorkflow):
    @classmethod
    def _build_algorithm(cls, config, *, param_dim, num_objectives, device, evaluate):
        return EvoXAlgorithmWrapper(
            MultipleEvaluationAlgorithm,
            dict(
                pop_size=config.pop_size,
                lb=torch.full((param_dim,), config.agent_network.lb, device=device),
                ub=torch.full((param_dim,), config.agent_network.ub, device=device),
                device=device,
            ),
            device=device,
            compile=config.torch_compile,
            evaluate=evaluate,
        )


@pytest.mark.integration
@pytest.mark.parametrize("jax_jit", [False, True])
@pytest.mark.parametrize(
    "objectives", [None, 1, 2], ids=["scalar", "vector1", "vector2"]
)
def test_common_workflow_counts_actual_evaluations_and_resumes(
    objectives, jit, jax_jit, tmp_path
):
    config = workflow_config()
    config.num_iters = 1
    base = get_class(config.workflow_cls) if objectives is None else NSGA2TestWorkflow
    cls = type("EvoXTestWorkflow", (base,), {})
    env = ToyEnv(2) if objectives is None else VectorRewardEnv(2, objectives)
    with patch(
        "evorl.algorithms.contrib.evox.evox_workflow.create_env", return_value=env
    ):
        config.torch_compile = jit
        workflow = cls.build_from_config(config, jit=jax_jit)
    workflow.recorder = Mock()
    workflow.checkpoint_manager = Mock()
    try:
        assert isinstance(workflow, Workflow)
        assert workflow.num_devices == 1
        assert hasattr(workflow._evaluate, "lower") is jax_jit
        assert hasattr(workflow.algorithm.evaluate_fn, "lower") is jax_jit
        assert not hasattr(base._evaluate, "lower")
        assert workflow.algorithm._compile is jit
        assert workflow.mesh is None
        assert workflow.config is not config
        assert OmegaConf.is_readonly(workflow.config)
        assert not OmegaConf.is_readonly(config)
        state = workflow.init(jax.random.key(5))
        snapshot = jax.tree.map(
            lambda x: np.array(x, copy=True), workflow_state_with_key_data(state)
        )
        # Initialization is complete before the first ordinary update.
        workflow.recorder.write.assert_not_called()
        workflow.checkpoint_manager.save.assert_not_called()
        assert int(state.metrics.iterations) == 0
        assert int(state.metrics.sampled_episodes) == 16
        second_metrics, second = workflow.step(state)
        chex.assert_trees_all_equal(workflow_state_with_key_data(state), snapshot)
        workflow.recorder.write.assert_not_called()
        workflow.checkpoint_manager.save.assert_not_called()
        # CSO evaluates only its two students after initialization.
        candidates = 2 if objectives is None else 4
        assert second_metrics.objectives.shape == (
            (candidates,) if objectives is None else (candidates, objectives)
        )
        assert int(second.metrics.iterations) == 1
        assert int(second.metrics.sampled_episodes) == 16 + 4 * candidates
        save(tmp_path / "workflow", state)
        restored = load(tmp_path / "workflow", state)
        resumed = workflow.learn(restored)
        chex.assert_trees_all_equal(
            workflow_state_with_key_data(resumed), workflow_state_with_key_data(second)
        )
        workflow.checkpoint_manager.save.assert_called_once()
        workflow.checkpoint_manager.save.assert_called_with(1, resumed, force=True)
        assert workflow.learn(resumed) is resumed
        logged = workflow.recorder.write.call_args.args[0]["objectives"]
        if objectives is not None:
            assert set(logged) == {f"objective_{i}" for i in range(objectives)}
    finally:
        workflow.close()


@pytest.mark.integration
@pytest.mark.parametrize("num_iters", [0, 1])
def test_workflow_initialization_respects_iteration_budget(num_iters):
    config = workflow_config()
    config.num_iters = num_iters
    cls = type("BudgetEvoXWorkflow", (get_class(config.workflow_cls),), {})
    with patch(
        "evorl.algorithms.contrib.evox.evox_workflow.create_env", return_value=ToyEnv(2)
    ):
        workflow = cls.build_from_config(config)
    workflow.recorder = Mock()
    workflow.checkpoint_manager = Mock()
    try:
        with patch.object(
            workflow.algorithm, "evaluate_fn", wraps=workflow.algorithm.evaluate_fn
        ) as evaluate:
            state = workflow.init(jax.random.key(8))
            assert int(state.metrics.iterations) == 0
            assert int(state.metrics.sampled_episodes) == 16
            assert evaluate.call_count == 1
            assert state.algorithm_state is not None
            workflow.recorder.write.assert_not_called()
            workflow.checkpoint_manager.save.assert_not_called()
            result = workflow.learn(state)
            assert int(result.metrics.iterations) == num_iters
            assert int(result.metrics.sampled_episodes) == 16 + 8 * num_iters
            assert evaluate.call_count == 1 + num_iters
            assert workflow.learn(result) is result
            assert evaluate.call_count == 1 + num_iters
            if num_iters:
                workflow.checkpoint_manager.save.assert_called_once_with(
                    1, result, force=True
                )
                assert workflow.recorder.write.call_count == 2
            else:
                assert result is state
                workflow.checkpoint_manager.save.assert_not_called()
                workflow.recorder.write.assert_not_called()
    finally:
        workflow.close()


def test_workflow_rejects_negative_budget_before_creating_environment():
    config = workflow_config()
    config.num_iters = -1
    cls = type("InvalidBudgetEvoXWorkflow", (get_class(config.workflow_cls),), {})
    with (
        patch("evorl.algorithms.contrib.evox.evox_workflow.create_env") as make_env,
        pytest.raises(ValueError, match="num_iters must be nonnegative"),
    ):
        cls.build_from_config(config)
    make_env.assert_not_called()


@pytest.mark.integration
def test_workflow_observation_warmup():
    config = workflow_config()
    config.normalize_obs = True
    config.random_timesteps = 4
    base = get_class(config.workflow_cls)
    cls = type("EvoXWarmupWorkflow", (base,), {})
    with (
        patch(
            "evorl.algorithms.contrib.evox.evox_workflow.create_env",
            return_value=ToyEnv(2),
        ),
        patch("evorl.algorithms.ec.obs_utils.create_env", return_value=ToyEnv(2)),
        patch(
            "evorl.algorithms.contrib.evox.evox_workflow.init_obs_preprocessor",
            wraps=init_obs_preprocessor,
        ) as initialize,
    ):
        workflow = cls.build_from_config(config)
        try:
            state = workflow.setup(jax.random.key(3))
            initialize.assert_called_once()
            assert not any(
                isinstance(value, jax.core.Tracer)
                for value in jax.tree.leaves(initialize.call_args.args)
            )
            assert (
                state.algorithm_state.evaluation_state.agent_state.obs_preprocessor_state.count
                > 0
            )
            metrics, _ = workflow.step(state)
            assert np.all(np.isfinite(metrics.objectives))
        finally:
            workflow.close()


def test_workflow_rejects_multi_device_and_mismatched_rewards():
    config = workflow_config()
    base = get_class(config.workflow_cls)
    cls = type("InvalidEvoXWorkflow", (base,), {})
    with (
        patch("evorl.algorithms.contrib.evox.evox_workflow.create_env") as make_env,
        pytest.raises(NotImplementedError, match="one device"),
    ):
        cls.build_from_config(config, multi_device=True)
    make_env.assert_not_called()
    with (
        patch(
            "evorl.algorithms.contrib.evox.evox_workflow.create_env",
            return_value=VectorRewardEnv(2),
        ),
        pytest.raises(ValueError, match="CSO requires scalar rewards"),
    ):
        cls.build_from_config(config)


@pytest.mark.integration
def test_workflow_checkpoint_schedule_and_resume(tmp_path):
    config = workflow_config()
    config.num_iters = 4
    config.output_dir = str(tmp_path)
    config.checkpoint.enable = True
    config.checkpoint.save_interval_steps = 3
    cls = type("CheckpointEvoXWorkflow", (get_class(config.workflow_cls),), {})
    with patch(
        "evorl.algorithms.contrib.evox.evox_workflow.create_env",
        return_value=ToyEnv(2),
    ):
        workflow = cls.build_from_config(config)
    workflow.recorder = Mock()
    try:
        state = workflow.learn(workflow.setup(jax.random.key(11)))
        workflow.checkpoint_manager.wait_until_finished()
        # Orbax saves the first step, interval steps and our forced final step.
        assert workflow.checkpoint_manager.all_steps() == [1, 3, 4]
        restored = workflow.checkpoint_manager.restore(4, state)
        _, expected = workflow.step(state)
        _, actual = workflow.step(restored)
        chex.assert_trees_all_equal(
            workflow_state_with_key_data(expected), workflow_state_with_key_data(actual)
        )
    finally:
        workflow.close()


@pytest.mark.integration
@pytest.mark.parametrize("objectives", [2, 3])
def test_rvea_workflow_keeps_int32_generation_state(objectives, jit):
    config = workflow_config("rvea")
    config.torch_compile = jit
    config.fr = 0.5
    cls = type("RVEATestWorkflow", (get_class(config.workflow_cls),), {})
    with patch(
        "evorl.algorithms.contrib.evox.evox_workflow.create_env",
        return_value=VectorRewardEnv(2, objectives),
    ):
        workflow = cls.build_from_config(config)
    try:
        state = workflow.init(jax.random.key(2))
        for generation in range(4):
            tensors = state.algorithm_state.params_and_buffers
            assert tensors["gen"].dtype == jnp.int32
            assert tensors["max_gen"].dtype == jnp.int32
            assert int(tensors["gen"]) == generation
            if generation < 3:
                _, state = workflow.step(state)
    finally:
        workflow.close()


@pytest.mark.parametrize("objectives", [None, 1])
def test_rvea_rejects_scalar_or_size_one_vector(objectives):
    config = workflow_config("rvea")
    base = get_class(config.workflow_cls)
    cls = type("InvalidRVEAWorkflow", (base,), {})
    env = ToyEnv(2) if objectives is None else VectorRewardEnv(2, objectives)
    with (
        patch(
            "evorl.algorithms.contrib.evox.evox_workflow.create_env", return_value=env
        ),
        pytest.raises(ValueError, match="at least two objectives"),
    ):
        cls.build_from_config(config)


@pytest.mark.integration
def test_workflow_counts_multiple_batches_and_advances_key_for_each():
    config = workflow_config()
    cls = type("MultipleEvaluationTestWorkflow", (MultipleEvaluationWorkflow,), {})
    with patch(
        "evorl.algorithms.contrib.evox.evox_workflow.create_env", return_value=ToyEnv(2)
    ):
        workflow = cls.build_from_config(config)
    try:
        state = workflow.setup(jax.random.key(4))
        assert int(state.metrics.iterations) == 0
        assert int(state.metrics.sampled_episodes) == 6 * 4
        initial_key = jax.random.split(jax.random.key(4), 4)[0]
        expected_initial_key = jax.random.split(jax.random.split(initial_key)[0])[0]
        chex.assert_trees_all_equal(
            jax.random.key_data(state.algorithm_state.evaluation_state.key),
            jax.random.key_data(expected_initial_key),
        )
        metrics, result = workflow.step(state)
        assert metrics.objectives.shape == (6,)
        assert int(result.metrics.iterations) == 1
        assert int(result.metrics.sampled_episodes) == 2 * 6 * 4
        expected_key = jax.random.split(
            jax.random.split(state.algorithm_state.evaluation_state.key)[0]
        )[0]
        chex.assert_trees_all_equal(
            jax.random.key_data(result.algorithm_state.evaluation_state.key),
            jax.random.key_data(expected_key),
        )
    finally:
        workflow.close()


@pytest.mark.integration
def test_workflow_evaluates_host_environments(host_envs):
    backend, _ = host_envs
    config = workflow_config()
    config.env = dict(env_name="Pendulum-v1", env_type=backend, max_episode_steps=2)
    base = get_class(config.workflow_cls)
    cls = type("HostEvoXWorkflow", (base,), {})
    workflow = cls.build_from_config(config)
    try:
        state = workflow.setup(jax.random.key(1))
        metrics, state = workflow.step(state)
        assert np.all(np.isfinite(metrics.objectives))
        assert int(state.metrics.sampled_episodes) == 24
    finally:
        workflow.close()
