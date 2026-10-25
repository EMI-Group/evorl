"""Replay prefill preserves data and starts training from reset environments."""

from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evorl.agent import RandomAgent
from evorl.algorithms.contrib.pop_td3 import PopTD3Workflow
from evorl.algorithms.meta.pbt_workflow import (
    PBTOffpolicyWorkflowTemplate,
    PBTWorkflowMetric,
)
from evorl.algorithms.offpolicy_utils import OffPolicyWorkflowTemplate
from evorl.distributed import META_POP_AXIS_NAME
from evorl.metrics import WorkflowMetric
from evorl.replay_buffers import ReplayBuffer
from evorl.types import State
from evorl.utils import running_statistics
from omegaconf import OmegaConf

from tests.helpers.envs import ToyEnv


@pytest.mark.parametrize(
    "random_timesteps,learning_start_timesteps,expected_samples,expected_resets",
    [(0, 0, 0, 0), (0, 4, 4, 2), (4, 4, 4, 2), (8, 4, 8, 2), (2, 5, 6, 3)],
    ids=[
        "empty",
        "policy_only",
        "random_only",
        "already_filled",
        "rounded_policy_phase",
    ],
)
def test_replay_warmup_skips_empty_phases(
    jit, random_timesteps, learning_start_timesteps, expected_samples, expected_resets
):
    workflow = object.__new__(OffPolicyWorkflowTemplate)
    workflow.config = OmegaConf.create(
        {
            "num_envs": 2,
            "random_timesteps": random_timesteps,
            "learning_start_timesteps": learning_start_timesteps,
        }
    )
    workflow.dp_axis_name = None
    workflow.env = ToyEnv(num_envs=2)
    env_state = workflow.env.reset(jax.random.key(15))
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
        env_state=env_state,
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
    np.testing.assert_array_equal(updated.env_state.env_state, 0)
    expected_key = jax.random.split(state.key, 3)[0]
    if expected_samples:
        expected_key = jax.random.split(expected_key)[0]
    np.testing.assert_array_equal(
        jax.random.key_data(updated.key), jax.random.key_data(expected_key)
    )


@pytest.mark.parametrize(
    "random_timesteps,learning_start_timesteps,expected_samples",
    [(0, 0, 0), (4, 4, 4), (2, 8, 8)],
    ids=["empty", "random_only", "both_phases"],
)
def test_pop_td3_warmup_resets_every_training_member(
    jit, random_timesteps, learning_start_timesteps, expected_samples
):
    workflow = object.__new__(PopTD3Workflow)
    workflow.config = OmegaConf.create(
        dict(
            num_envs=2,
            pop_size=3,
            random_timesteps=random_timesteps,
            learning_start_timesteps=learning_start_timesteps,
        )
    )
    workflow.dp_axis_name = None
    workflow.env = ToyEnv(num_envs=2)
    workflow.agent = RandomAgent()
    workflow.replay_buffer = ReplayBuffer(capacity=16, sample_batch_size=1)
    keys = jax.random.split(jax.random.key(16), 3)
    state = State(
        key=jax.random.key(17),
        env_state=jax.vmap(workflow.env.reset)(keys),
        agent_state=jax.vmap(workflow.agent.init, in_axes=(None, None, 0))(
            workflow.env.obs_space, workflow.env.action_space, keys
        ),
        replay_buffer_state=workflow._setup_replaybuffer(jax.random.key(18)),
        metrics=WorkflowMetric(),
    )
    prefill = workflow._postsetup_replaybuffer
    updated = (jax.jit(prefill) if jit else prefill)(state)
    np.testing.assert_array_equal(updated.env_state.env_state, np.zeros((3, 2)))
    assert int(updated.replay_buffer_state.buffer_size) == expected_samples
    assert int(updated.metrics.sampled_timesteps) == expected_samples


@pytest.mark.integration
@pytest.mark.parametrize("nested", [False, True], ids=["single", "nested"])
def test_host_warmup_resets_actual_simulators_before_training(monkeypatch, nested):
    gym = pytest.importorskip("gymnasium")
    from evorl.envs.gymnasium import GymnasiumAdapter

    class CounterEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 1000, (3,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.count = 0
            return np.zeros(3, np.float32), {}

        def step(self, action):
            self.count += 1
            return np.full(3, self.count, np.float32), 1.0, False, False, {}

    def make_vec(name, num_envs, vector_kwargs, **kwargs):
        return gym.vector.SyncVectorEnv(
            [CounterEnv] * num_envs,
            autoreset_mode=vector_kwargs["autoreset_mode"],
        )

    monkeypatch.setattr(gym, "make_vec", make_vec)
    workflow = object.__new__(OffPolicyWorkflowTemplate)
    workflow.config = OmegaConf.create(
        dict(num_envs=2, random_timesteps=4, learning_start_timesteps=6)
    )
    workflow.dp_axis_name = None
    workflow.env = GymnasiumAdapter(
        "Counter",
        1000,
        2,
        record_ori_obs=True,
        discount=1.0,
        vecenv_kwargs=dict(autoreset_mode=gym.vector.AutoresetMode.SAME_STEP),
    )
    workflow.agent = RandomAgent()
    workflow.replay_buffer = ReplayBuffer(capacity=16, sample_batch_size=1)

    def run(key):
        state = State(
            key=key,
            env_state=workflow.env.reset(key),
            agent_state=workflow.agent.init(
                workflow.env.obs_space, workflow.env.action_space, key
            ),
            replay_buffer_state=workflow._setup_replaybuffer(key),
            metrics=WorkflowMetric(),
        )
        state = workflow._postsetup_replaybuffer(state)
        # The first training action must advance a new native episode to step 1.
        next_env = workflow.env.step(state.env_state, jnp.zeros((2, 1)))
        return state, next_env

    try:
        keys = jax.random.key(19)
        batch_shape = ()
        if nested:
            keys = jax.random.split(keys, (2, 2))
            batch_shape = (2, 2)
            run = jax.vmap(jax.vmap(run))
        state, next_env = jax.jit(run)(keys)
        jax.block_until_ready(next_env)
        np.testing.assert_array_equal(state.env_state.obs, 0)
        np.testing.assert_array_equal(next_env.obs, 1)
        np.testing.assert_array_equal(state.replay_buffer_state.buffer_size, 6)
        np.testing.assert_array_equal(state.metrics.sampled_timesteps, 6)
        assert workflow.env.env.num_envs == 2 * int(np.prod(batch_shape))
    finally:
        jax.effects_barrier()
        workflow.env.env.close()


@pytest.mark.parametrize(
    "normalize_obs", [False, True], ids=["raw_obs", "normalize_obs"]
)
def test_pbt_warmup_preserves_population_observation_statistics(
    jit, normalize_obs, mesh_factory
):
    target = object.__new__(OffPolicyWorkflowTemplate)
    target.env = ToyEnv(num_envs=2)
    target.replay_buffer = ReplayBuffer(capacity=16, sample_batch_size=1)
    workflow = object.__new__(PBTOffpolicyWorkflowTemplate)
    workflow.workflow = target
    workflow.replay_buffer = target.replay_buffer
    workflow.config = OmegaConf.create(
        {
            "target_workflow": {"num_envs": 2},
            "random_timesteps": 4,
            "shared_replay_buffer": True,
        }
    )
    agent_state = RandomAgent().init(
        target.env.obs_space, target.env.action_space, jax.random.key(14)
    )
    if normalize_obs:
        agent_state = agent_state.replace(
            obs_preprocessor_state=running_statistics.init_state(jnp.zeros(3))
        )
    population = jax.vmap(
        lambda key: State(
            key=key, agent_state=agent_state, env_state=target.env.reset(key)
        )
    )(jax.random.split(jax.random.key(15), 4))
    mesh = mesh_factory(names=(META_POP_AXIS_NAME,))
    with jax.set_mesh(mesh):
        state = State(
            key=jax.device_put(jax.random.key(12), jax.P()),
            pop_workflow_state=jax.device_put(population, jax.P(META_POP_AXIS_NAME)),
            replay_buffer_state=jax.device_put(
                target._setup_replaybuffer(jax.random.key(13)), jax.P()
            ),
            metrics=jax.device_put(PBTWorkflowMetric(), jax.P()),
        )
        warmup = workflow._postsetup_replaybuffer
        updated = (jax.jit(warmup) if jit else warmup)(state)

    assert int(updated.replay_buffer_state.buffer_size) == 4
    assert int(state.replay_buffer_state.buffer_size) == 0
    np.testing.assert_array_equal(
        updated.pop_workflow_state.env_state.env_state, np.zeros((4, 2))
    )
    np.testing.assert_allclose(updated.metrics.sampled_timesteps_m, 4e-6)
    stats = updated.pop_workflow_state.agent_state.obs_preprocessor_state
    if normalize_obs:
        observations = np.asarray(updated.replay_buffer_state.data.obs[:4])
        np.testing.assert_array_equal(stats.count, [4] * 4)
        np.testing.assert_allclose(
            stats.mean, np.broadcast_to(observations.mean(0), (4, 3)), rtol=1e-6
        )
        np.testing.assert_allclose(
            stats.std, np.broadcast_to(observations.std(0), (4, 3)), rtol=1e-5
        )
        assert stats.count.sharding.spec == jax.P(META_POP_AXIS_NAME)
        np.testing.assert_array_equal(
            state.pop_workflow_state.agent_state.obs_preprocessor_state.count, [0] * 4
        )
    else:
        assert stats is None
