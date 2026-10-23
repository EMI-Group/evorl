"""Joint trajectory boundaries and terminal observations across autoresets."""

import jax
import jax.numpy as jnp
import numpy as np

from evorl.envs.wrappers import EpisodeWrapper, VmapAutoResetWrapper
from evorl.rollout import multi_agent_rollout
from evorl.types import PyTreeDict
from tests.helpers.multi_agent_envs import ToyMultiAgentEnv


def test_joint_rollout_retains_terminal_observations_and_activity(jit):
    env = VmapAutoResetWrapper(EpisodeWrapper(ToyMultiAgentEnv(), 3), 2)

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
            env, actions, env.reset(reset_key), None, rollout_key, 6
        )

    trajectory, state = (jax.jit(collect) if jit else collect)(jax.random.key(0))
    np.testing.assert_array_equal(
        trajectory.extras.active.first[:, 0], [True, True, False] * 2
    )
    np.testing.assert_array_equal(trajectory.extras.active.second, True)
    np.testing.assert_array_equal(trajectory.dones.first[:, 0], [0, 1, 1] * 2)
    np.testing.assert_array_equal(trajectory.dones.second[:, 0], [0, 0, 1] * 2)
    np.testing.assert_array_equal(trajectory.extras.terminations.second, 0)
    np.testing.assert_array_equal(
        trajectory.next_obs.first.observation[:, 0, 0], [1, 2, 3] * 2
    )
    np.testing.assert_array_equal(
        trajectory.obs.first.observation[:, 0, 0], [0, 1, 2] * 2
    )
    np.testing.assert_array_equal(
        trajectory.extras.next_global_obs[:, 0, 0], [1, 2, 3] * 2
    )
    np.testing.assert_array_equal(state.obs.first.observation[:, 0], 0)
