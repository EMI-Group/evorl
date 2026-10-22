"""Real Brax policy optimization, logging and checkpoint replay."""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf

torch = pytest.importorskip("torch", minversion="2.9.0")
pytest.importorskip("evox")

from evorl.recorders import LogRecorder
from evorl.utils.orbax_utils import DummyCheckpointManager
from tests.helpers.evox import (
    NSGA2TestWorkflow,
    workflow_config,
    workflow_state_with_key_data,
    rng_states,
    assert_rng_states_equal,
)


@pytest.mark.integration
@pytest.mark.parametrize("torch_compile", [False, True])
def test_evox_trains_real_brax_scalar_and_vector_rewards(torch_compile):
    root = Path(__file__).resolve().parents[2]
    for vector in (False, True):
        config = workflow_config()
        config.torch_compile = torch_compile
        config.env = OmegaConf.load(
            root / f"configs/env/brax/{'ant_multi_objective' if vector else 'ant'}.yaml"
        )
        config.env.max_episode_steps = 2
        config.num_envs = 1
        config.episodes_for_fitness = 1
        base = NSGA2TestWorkflow if vector else get_class(config.workflow_cls)
        cls = type("BraxEvoXWorkflow", (base,), {})
        workflow = cls.build_from_config(config)
        try:
            state = workflow.setup(jax.random.key(2))
            metrics, state = workflow.step(state)
            assert np.all(np.isfinite(metrics.objectives))
            assert int(state.metrics.iterations) == 1
        finally:
            workflow.close()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ["cso", "rvea"])
@pytest.mark.parametrize("torch_compile", [False, True])
def test_evox_brax_full_training_and_checkpoint_resume(
    algorithm, torch_compile, tmp_path
):
    root = Path(__file__).resolve().parents[2]
    env_name = "ant" if algorithm == "cso" else "ant_multi_objective"
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / f"configs/agent/contrib/{algorithm}-evox.yaml"),
        {
            "env": OmegaConf.load(root / f"configs/env/brax/{env_name}.yaml"),
            "num_iters": 5,
            "pop_size": 16,
            "num_envs": 4,
            # Exercise rounding to a full batch of four episodes.
            "episodes_for_fitness": 3,
            "torch_compile": torch_compile,
            "output_dir": str(tmp_path),
            "checkpoint": {"enable": True, "save_interval_steps": 2},
        },
    )
    if algorithm == "rvea":
        # Cover both branches of native reference-vector adaptation.
        config.fr = 0.5
    cls = type("FullBraxEvoXWorkflow", (get_class(config.workflow_cls),), {})
    before = rng_states()
    workflow = cls.build_from_config(config)
    log_path = tmp_path / "training.log"
    workflow.add_recorders([LogRecorder(str(log_path), console=False)])
    try:
        state = workflow.init(jax.random.key(42))
        assert int(state.metrics.iterations) == 0
        assert int(state.metrics.sampled_episodes) == 16 * 4
        assert log_path.read_text() == ""
        assert workflow.checkpoint_manager.all_steps() == []
        observation_statistics = (
            state.algorithm_state.evaluation_state.agent_state.obs_preprocessor_state
        )
        assert int(observation_statistics.count) == config.random_timesteps
        final = workflow.learn(state)
        assert int(final.metrics.iterations) == 5
        candidates_per_update = 8 if algorithm == "cso" else 16
        assert int(final.metrics.sampled_episodes) == 4 * (
            16 + 5 * candidates_per_update
        )
        assert float(final.metrics.sampled_timesteps_m) > 0
        tensors = final.algorithm_state.params_and_buffers
        assert all(value.device == jax.local_devices()[0] for value in tensors.values())
        population, fitness = np.asarray(tensors["pop"]), np.asarray(tensors["fit"])
        if algorithm == "cso":
            assert np.all(np.isfinite(population))
            assert np.all(np.isfinite(fitness))
        else:
            # RVEA reserves NaN rows for reference vectors without survivors.
            valid = ~np.isnan(population).any(axis=-1)
            assert valid.any()
            assert np.all(np.isfinite(population[valid]))
            assert np.all(np.isfinite(fitness[valid]))
            assert np.array_equal(~valid, np.isnan(fitness).all(axis=-1))
            assert int(tensors["gen"]) == 5
            assert tensors["gen"].dtype == jnp.int32
            assert tensors["max_gen"].dtype == jnp.int32
        text = log_path.read_text()
        assert text.count("iteration ") == 10
        for iteration in range(1, 6):
            assert text.count(f"iteration {iteration}:") == 2
        if algorithm == "cso":
            assert "objective_" not in text
        else:
            assert "reward_forward" in text
            assert "reward_ctrl" in text
        workflow.checkpoint_manager.wait_until_finished()
        assert workflow.checkpoint_manager.all_steps() == [1, 2, 4, 5]
        restored = workflow.checkpoint_manager.restore(2, final)
        workflow.checkpoint_manager.close()
        workflow.checkpoint_manager = DummyCheckpointManager()
        resumed = workflow.learn(restored)
        chex.assert_trees_all_equal(
            workflow_state_with_key_data(resumed), workflow_state_with_key_data(final)
        )
        assert workflow.learn(resumed) is resumed
        assert_rng_states_equal(before)
    finally:
        workflow.close()
