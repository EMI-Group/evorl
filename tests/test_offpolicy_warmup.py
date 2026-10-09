"""Exercise empty and already-completed replay-buffer warmup phases."""

from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from tests._workflow_multidevice_cases import ToyEnv
from evorl.agent import RandomAgent
from evorl.algorithms.offpolicy_utils import OffPolicyWorkflowTemplate
from evorl.metrics import WorkflowMetric
from evorl.replay_buffers import ReplayBuffer
from evorl.types import State
from evorl.utils import running_statistics


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(
    "random_timesteps,learning_start_timesteps,expected_samples,expected_resets",
    [(0, 0, 0, 0), (0, 4, 4, 1), (4, 4, 4, 1), (8, 4, 8, 1), (2, 5, 6, 2)],
)
def test_replay_warmup_skips_empty_phases(
    jit, random_timesteps, learning_start_timesteps, expected_samples, expected_resets
):
    workflow = object.__new__(OffPolicyWorkflowTemplate)
    workflow.config = OmegaConf.create(
        dict(
            num_envs=2,
            random_timesteps=random_timesteps,
            learning_start_timesteps=learning_start_timesteps,
        )
    )
    workflow.dp_axis_name = None
    workflow.env = ToyEnv(num_envs=2)
    workflow.env.reset = Mock(wraps=workflow.env.reset)
    workflow.agent = RandomAgent()
    workflow.replay_buffer = ReplayBuffer(capacity=16, sample_batch_size=1)
    state = State(
        key=jax.random.key(12),
        agent_state=workflow.agent.init(
            workflow.env.obs_space, workflow.env.action_space, jax.random.key(14)
        ).replace(
            obs_preprocessor_state=running_statistics.init_state(jnp.zeros(3)),
        ),
        replay_buffer_state=workflow._setup_replaybuffer(jax.random.key(13)),
        metrics=WorkflowMetric(sampled_timesteps=jnp.uint32(10)),
    )
    warmup = workflow._postsetup_replaybuffer
    if jit:
        warmup = jax.jit(warmup)
    updated = warmup(state)
    assert workflow.env.reset.call_count == expected_resets
    assert int(updated.metrics.sampled_timesteps) == 10 + expected_samples
    assert int(updated.replay_buffer_state.buffer_size) == expected_samples
    assert int(updated.replay_buffer_state.current_index) == expected_samples
    assert float(updated.agent_state.obs_preprocessor_state.count) == expected_samples
    assert int(state.replay_buffer_state.buffer_size) == 0
    assert int(state.metrics.sampled_timesteps) == 10
    np.testing.assert_array_equal(
        jax.random.key_data(updated.key),
        jax.random.key_data(jax.random.split(state.key, 3)[0]),
    )
