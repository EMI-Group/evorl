"""Check that ES persists its final state outside the save interval."""

from unittest.mock import Mock

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import orbax.checkpoint as ocp
import pytest

from evorl.algorithms.ec.so.es_workflow import ESWorkflowTemplate
from evorl.metrics import ECTrainMetric, ECWorkflowMetric, EvaluateMetric
from evorl.types import PyTreeDict, State
from evorl.utils.orbax_utils import CheckpointManager


class _CountingESWorkflow(ESWorkflowTemplate):
    def step(self, state):
        return ECTrainMetric(
            objectives=jnp.ones(2), ec_metrics=PyTreeDict()
        ), state.replace(
            metrics=state.metrics.replace(iterations=state.metrics.iterations + 1)
        )

    def evaluate(self, state):
        return EvaluateMetric(
            episode_returns=jnp.float32(0), episode_lengths=jnp.float32(0)
        ), state


@pytest.mark.parametrize("start_iteration", [0, 1])
def test_es_saves_final_checkpoint_between_save_intervals(tmp_path, start_iteration):
    workflow = object.__new__(_CountingESWorkflow)
    workflow.config = OmegaConf.create(dict(num_iters=3, eval_interval=10))
    workflow.recorder = Mock()
    mesh = jax.make_mesh(
        (1,),
        ("devices",),
        axis_types=(jax.sharding.AxisType.Explicit,),
        devices=jax.local_devices()[:1],
    )
    with jax.set_mesh(mesh):
        state = State(
            metrics=ECWorkflowMetric(
                best_objective=jnp.float32(0), iterations=jnp.uint32(start_iteration)
            )
        )
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=10, enable_async_checkpointing=False
        )
        with CheckpointManager(tmp_path / "checkpoints", options=options) as manager:
            workflow.checkpoint_manager = manager
            workflow.learn(state)
            assert manager.latest_step() == 3
            restored = manager.restore(3, state)
            assert int(restored.metrics.iterations) == 3
