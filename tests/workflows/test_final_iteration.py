"""Check final evaluation and forced saves when several steps run per loop."""

from unittest.mock import Mock

import jax.numpy as jnp
import pytest
from evorl.algorithms.contrib.pop_td3 import PopTD3Workflow
from evorl.algorithms.contrib.td3_v2 import TD3V2Workflow
from evorl.algorithms.contrib.td3_v3 import TD3V3Workflow
from evorl.algorithms.offpolicy_utils import OffPolicyWorkflowTemplate
from evorl.algorithms.td7 import TD7Workflow
from evorl.metrics import EvaluateMetric, TrainMetric, WorkflowMetric
from evorl.types import State
from omegaconf import OmegaConf


class _CountingWorkflow(OffPolicyWorkflowTemplate):
    def step(self, state):
        return TrainMetric(loss=jnp.float32(0)), state.replace(
            metrics=state.metrics.replace(
                iterations=state.metrics.iterations + 1,
                sampled_timesteps=state.metrics.sampled_timesteps + 1,
                sampled_episodes=state.metrics.sampled_episodes + 1,
            )
        )


@pytest.mark.parametrize(
    "learn",
    [
        OffPolicyWorkflowTemplate.learn,
        TD3V2Workflow.learn,
        TD3V3Workflow.learn,
        PopTD3Workflow.learn,
        TD7Workflow.learn,
    ],
    ids=["template", "td3-v2", "td3-v3", "pop-td3", "td7"],
)
@pytest.mark.parametrize("fold_iters", [1, 2, 3])
@pytest.mark.parametrize("start_iteration", [0, 5])
def test_folded_steps_evaluate_and_force_save_at_the_end(
    learn, fold_iters, start_iteration
):
    workflow = object.__new__(_CountingWorkflow)
    workflow.mesh = None
    workflow.config = OmegaConf.create(
        {
            "rollout_length": 1,
            "num_envs": 1,
            "pop_size": 1,
            "fold_iters": fold_iters,
            "total_timesteps": 10 + 3 * fold_iters,
            "rollout_episodes": 1,
            "total_episodes": 10 + 3 * fold_iters,
            "eval_interval": 100,
            "save_replay_buffer": True,
        }
    )
    workflow.recorder = Mock()
    workflow.checkpoint_manager = Mock()
    workflow.evaluate = Mock(
        side_effect=lambda state: (
            EvaluateMetric(
                episode_return=jnp.float32(0), episode_length=jnp.float32(0)
            ),
            state,
        )
    )
    state = State(
        metrics=WorkflowMetric(
            iterations=jnp.uint32(start_iteration),
            sampled_timesteps=jnp.uint32(10),
            sampled_episodes=jnp.uint32(10),
        )
    )

    updated = learn(workflow, state)

    expected_iteration = start_iteration + 3 * fold_iters
    assert int(updated.metrics.iterations) == expected_iteration
    workflow.evaluate.assert_called_once()
    assert (
        int(workflow.evaluate.call_args.args[0].metrics.iterations)
        == expected_iteration
    )
    saves = workflow.checkpoint_manager.save.call_args_list
    assert len(saves) == 3
    assert all(not call.kwargs["force"] for call in saves[:-1])
    assert saves[-1].args[0] == expected_iteration
    assert saves[-1].kwargs["force"]
    for call in saves:
        assert call.args[0] == int(call.args[1].metrics.iterations)
