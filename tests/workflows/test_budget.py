"""Check training budgets against the devices used by each workflow."""

from unittest.mock import Mock

import jax
import jax.numpy as jnp
import pytest
from evorl.algorithms.a2c import A2CWorkflow
from evorl.algorithms.contrib.a2c_v2 import A2CWorkflow as A2CV2Workflow
from evorl.algorithms.contrib.pop_td3 import PopTD3Workflow
from evorl.algorithms.contrib.td3_onpolicy import TD3OnPolicyWorkflow
from evorl.algorithms.contrib.td3_v2 import TD3V2Workflow
from evorl.algorithms.contrib.td3_v3 import TD3V3Workflow
from evorl.algorithms.impala import IMPALAWorkflow
from evorl.algorithms.offpolicy_utils import OffPolicyWorkflowTemplate
from evorl.algorithms.ppo import PPOWorkflow
from evorl.algorithms.td7 import TD7Workflow
from evorl.metrics import EvaluateMetric, TrainMetric, WorkflowMetric
from evorl.types import State
from omegaconf import OmegaConf


class _CountingWorkflow(OffPolicyWorkflowTemplate):
    def step(self, state):
        return TrainMetric(train_episode_return=jnp.float32(0)), state.replace(
            metrics=state.metrics.replace(
                iterations=state.metrics.iterations + 1,
                sampled_timesteps=state.metrics.sampled_timesteps
                + self.samples_per_step,
                sampled_episodes=state.metrics.sampled_episodes
                + self.episodes_per_step,
            )
        )


@pytest.mark.parametrize("mesh_size", [None, 1, 2], ids=["single", "mesh-1", "mesh-2"])
@pytest.mark.parametrize(
    "learn",
    [
        PPOWorkflow.learn,
        A2CWorkflow.learn,
        IMPALAWorkflow.learn,
        A2CV2Workflow.learn,
        TD3OnPolicyWorkflow.learn,
        OffPolicyWorkflowTemplate.learn,
        TD3V2Workflow.learn,
        TD3V3Workflow.learn,
        PopTD3Workflow.learn,
        TD7Workflow.learn,
    ],
    ids=[
        "ppo",
        "a2c",
        "impala",
        "a2c-v2",
        "td3-onpolicy",
        "offpolicy",
        "td3-v2",
        "td3-v3",
        "pop-td3",
        "td7",
    ],
)
def test_learn_uses_only_workflow_devices(learn, mesh_size):
    if jax.local_device_count() < 2:
        pytest.skip("Run with JAX_NUM_CPU_DEVICES=2 to check unused visible devices")

    workflow = object.__new__(_CountingWorkflow)
    workflow.mesh = (
        None
        if mesh_size is None
        else jax.make_mesh(
            (mesh_size,),
            ("devices",),
            axis_types=(jax.sharding.AxisType.Explicit,),
            devices=jax.local_devices()[:mesh_size],
        )
    )
    workers = mesh_size or 1
    workflow.samples_per_step = 4 * workers
    workflow.episodes_per_step = workers
    workflow.config = OmegaConf.create(
        {
            "num_envs": 2,
            "rollout_length": 2,
            "fold_iters": 2,
            "pop_size": 1,
            "total_timesteps": 48,
            "rollout_episodes": 1,
            "total_episodes": 12,
            "eval_interval": 2,
            "save_replay_buffer": True,
        }
    )
    workflow.recorder = Mock()
    workflow.checkpoint_manager = Mock()
    workflow.evaluate = Mock(
        side_effect=lambda state: (
            EvaluateMetric(
                episode_returns=jnp.float32(0), episode_lengths=jnp.float32(0)
            ),
            state,
        )
    )

    updated = learn(workflow, State(metrics=WorkflowMetric()))

    assert int(updated.metrics.sampled_timesteps) == 48
    assert int(updated.metrics.iterations) == 12 // workers
    save = workflow.checkpoint_manager.save.call_args
    assert save.args[0] == int(updated.metrics.iterations)
    assert save.kwargs["force"]
