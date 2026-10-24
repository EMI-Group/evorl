"""Joint trajectory boundaries and terminal observations across autoresets."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.envs.wrappers import (
    EpisodeWrapper,
    VmapAutoResetWrapper,
    VmapEnvPoolAutoResetWrapper,
)
from evorl.rollout import multi_agent_rollout
from evorl.types import PyTreeDict
from tests.helpers.multi_agent_envs import ToyMultiAgentEnv


@pytest.mark.parametrize("envpool", [False, True])
def test_joint_rollout_retains_terminal_observations_and_activity(jit, envpool):
    wrapper = VmapEnvPoolAutoResetWrapper if envpool else VmapAutoResetWrapper
    env = wrapper(EpisodeWrapper(ToyMultiAgentEnv(), 3, record_ori_obs=not envpool), 2)

    def actions(agent_state, batch, key):
        return PyTreeDict(
            {
                name: jnp.argmax(obs.action_mask, axis=-1)
                for name, obs in batch.obs.items()
            }
        ), PyTreeDict()

    def collect(key):
        reset_key, rollout_key = jax.random.split(key)
        return multi_agent_rollout(
            env, actions, env.reset(reset_key), None, rollout_key, 8 if envpool else 6
        )

    trajectory, state = (jax.jit(collect) if jit else collect)(jax.random.key(0))
    np.testing.assert_array_equal(
        trajectory.extras.active.first[:, 0],
        ([True, True, False, False] if envpool else [True, True, False]) * 2,
    )
    np.testing.assert_array_equal(
        trajectory.extras.active.second[:, 0],
        ([True, True, True, False] if envpool else [True, True, True]) * 2,
    )
    np.testing.assert_array_equal(
        trajectory.dones.first[:, 0], ([0, 1, 1, 0] if envpool else [0, 1, 1]) * 2
    )
    np.testing.assert_array_equal(
        trajectory.dones.second[:, 0], ([0, 0, 1, 0] if envpool else [0, 0, 1]) * 2
    )
    np.testing.assert_array_equal(trajectory.extras.terminations.second, 0)
    np.testing.assert_array_equal(
        trajectory.next_obs.first.observation[:, 0, 0],
        ([1, 2, 3, 0] if envpool else [1, 2, 3]) * 2,
    )
    np.testing.assert_array_equal(
        trajectory.obs.first.observation[:, 0, 0],
        ([0, 1, 2, 3] if envpool else [0, 1, 2]) * 2,
    )
    np.testing.assert_array_equal(
        trajectory.extras.next_global_obs[:, 0, 0],
        ([1, 2, 3, 0] if envpool else [1, 2, 3]) * 2,
    )
    np.testing.assert_array_equal(state.obs.first.observation[:, 0], 0)
