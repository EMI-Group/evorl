"""Run native CMA-ES, SepCMA-ES and vector-reward NSGA-II workflows."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import jax
import numpy as np
import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf

from tests.helpers.envs import ToyEnv, VectorRewardEnv

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("agent", ["cmaes", "sepcmaes", "nsga2"])
@pytest.mark.parametrize("multi_device", [False, True])
def test_native_ec_workflow_setup_step_and_learn(agent, multi_device, jit):
    root = Path(__file__).resolve().parents[2]
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / f"configs/agent/ec/{agent}.yaml"),
        {
            "env": {"max_episode_steps": 2},
            "num_iters": 2,
            "pop_size": 4,
            "num_elites": 2,
            "num_envs": 2,
            "episodes_for_fitness": 2,
            "num_eval_envs": 2,
            "eval_episodes": 2,
            "normalize_obs": False,
            "agent_network": {"actor_hidden_layer_sizes": [2]},
        },
    )
    base = get_class(config.workflow_cls)
    workflow_cls = type("NativeECTestWorkflow", (base,), {})
    env_cls = VectorRewardEnv if agent == "nsga2" else ToyEnv
    with patch(
        f"{base.__module__}.create_env",
        side_effect=lambda _, parallel, **kw: env_cls(parallel),
    ):
        workflow = workflow_cls.build_from_config(
            config, multi_device=multi_device, jit=jit
        )
    workflow.checkpoint_manager = Mock()
    workflow.recorder = Mock()
    try:
        with jax.set_mesh(workflow.mesh) if multi_device else nullcontext():
            state = workflow.init(jax.random.key(42))
            metrics, stepped = workflow.step(state)
            jax.block_until_ready(stepped)
            assert int(stepped.metrics.iterations) == 1
            assert int(stepped.metrics.sampled_episodes) == 8
            assert metrics.objectives.shape == ((4, 2) if agent == "nsga2" else (4,))
            assert np.all(np.isfinite(metrics.objectives))
            if agent != "nsga2":
                evaluated, _ = workflow.evaluate(stepped)
                assert np.isfinite(evaluated.episode_returns)
            final = workflow.learn(state)
            jax.block_until_ready(final)
            assert int(final.metrics.iterations) == 2
            assert int(final.metrics.sampled_episodes) == 16
            assert workflow.checkpoint_manager.save.call_args.kwargs["force"] is True
            if agent == "nsga2":
                logged = workflow.recorder.write.call_args.args[0]
                assert set(logged["objectives"]) == {"objective_0", "objective_1"}
            workflow.checkpoint_manager.reset_mock()
            assert workflow.learn(final) is final
            workflow.checkpoint_manager.save.assert_not_called()
    finally:
        workflow.close()


@pytest.mark.parametrize(
    "num_objectives", [None, 1, 3], ids=["scalar", "vector1", "vector3"]
)
def test_nsga2_infers_native_reward_shape_without_reward_keys(num_objectives):
    root = Path(__file__).resolve().parents[2]
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / "configs/agent/ec/nsga2.yaml"),
        {
            "env": {"max_episode_steps": 2},
            "num_iters": 1,
            "pop_size": 4,
            "num_envs": 2,
            "episodes_for_fitness": 2,
            "normalize_obs": False,
            "agent_network": {"actor_hidden_layer_sizes": [2]},
        },
    )
    base = get_class(config.workflow_cls)
    workflow_cls = type("VectorNSGA2Workflow", (base,), {})
    env = ToyEnv(2) if num_objectives is None else VectorRewardEnv(2, num_objectives)
    with patch(f"{base.__module__}.create_env", return_value=env):
        if num_objectives is None or num_objectives < 2:
            with pytest.raises(ValueError, match="requires vector rewards"):
                workflow_cls.build_from_config(config)
            return
        workflow = workflow_cls.build_from_config(config)
    workflow.recorder = Mock()
    workflow.checkpoint_manager = Mock()
    try:
        state = workflow.setup(jax.random.key(3))
        final = workflow.learn(state)
        assert final.ec_opt_state.fitnesses.shape == (4, num_objectives)
        logged = workflow.recorder.write.call_args.args[0]
        assert set(logged["objectives"]) == {
            f"objective_{i}" for i in range(num_objectives)
        }
    finally:
        workflow.close()


def test_nsga2_trains_with_real_brax_vector_rewards():
    root = Path(__file__).resolve().parents[2]
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / "configs/agent/ec/nsga2.yaml"),
        {
            "env": OmegaConf.load(root / "configs/env/brax/ant_multi_objective.yaml"),
            "num_iters": 2,
            "pop_size": 4,
            "num_envs": 1,
            "episodes_for_fitness": 1,
            "normalize_obs": False,
            "agent_network": {"actor_hidden_layer_sizes": [2]},
        },
    )
    config.env.max_episode_steps = 2
    base = get_class(config.workflow_cls)
    workflow_cls = type("BraxNSGA2TestWorkflow", (base,), {})
    workflow = workflow_cls.build_from_config(config)
    try:
        state = workflow.setup(jax.random.key(1))
        for _ in range(2):
            metrics, state = workflow.step(state)
            assert metrics.objectives.shape == (4, 2)
            assert np.all(np.isfinite(metrics.objectives))
        assert int(state.metrics.iterations) == 2
        assert int(state.metrics.sampled_episodes) == 8
        assert np.all(np.isfinite(state.ec_opt_state.fitnesses))
    finally:
        workflow.close()
