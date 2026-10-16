"""Exercise real PBT targets on the independently named meta-population axis."""

from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evorl.algorithms.meta.pbt_workflow import PBTOffpolicyWorkflowTemplate
from evorl.algorithms.offpolicy_utils import OffPolicyWorkflowTemplate
from evorl.distributed import META_POP_AXIS_NAME, POP_AXIS_NAME
from evorl.replay_buffers import ReplayBuffer
from evorl.rollout import eval_rollout_episode, fast_eval_rollout_episode, rollout
from evorl.types import PyTreeDict
from hydra.utils import get_class
from omegaconf import OmegaConf

from tests.helpers.envs import ToyEnv
from tests.helpers.sharding import compiled_collectives

pytestmark = pytest.mark.integration


def test_shared_replay_buffer_is_replicated_and_accepts_writes(meta_mesh, jit):
    target = object.__new__(OffPolicyWorkflowTemplate)
    target.env = ToyEnv(2)
    target.replay_buffer = ReplayBuffer(capacity=16, sample_batch_size=2)
    workflow = object.__new__(PBTOffpolicyWorkflowTemplate)
    workflow.workflow = target
    key = jax.device_put(jax.random.key(3), jax.P())
    initialize = workflow._setup_replaybuffer
    if jit:
        initialize = jax.jit(initialize)
    state = initialize(key)
    assert int(state.current_index) == 0
    assert int(state.buffer_size) == 0
    assert state.data.obs.shape == (16, 3)
    for leaf in jax.tree.leaves(state):
        assert leaf.sharding.is_fully_replicated
        assert len(leaf.addressable_shards) == 2

    samples = jax.tree.map(
        lambda x: jnp.ones((2, *x.shape[1:]), dtype=x.dtype), state.data
    )
    updated = target.replay_buffer.add(state, samples)
    assert int(updated.buffer_size) == 2
    assert int(updated.current_index) == 2
    for leaf in jax.tree.leaves(updated.data):
        for shard in leaf.addressable_shards:
            written = np.asarray(shard.data)[:2]
            np.testing.assert_array_equal(written, np.ones_like(written))
    compiled = jax.jit(workflow._setup_replaybuffer).lower(key).compile()
    assert not compiled_collectives(compiled.as_text())


@pytest.mark.parametrize("rollout_kind", ["train", "episode", "fast"])
def test_rollout_uses_local_population_axes(rollout_kind, meta_mesh):
    class VariableLengthEnv(ToyEnv):
        def step(self, state, action):
            next_state = super().step(state, action)
            # Input-dependent termination keeps the evaluation loop dynamic;
            # uniform lengths can hide automatic cross-device loop reductions.
            threshold = 1 + jnp.floor(state.obs[:, 0] * 4).astype(jnp.int32)
            return next_state.replace(
                done=(next_state.env_state >= threshold).astype(jnp.float32)
            )

    env = VariableLengthEnv(2)

    def action_fn(agent_state, batch, key):
        return batch.obs[:, :1], PyTreeDict()

    def instance(key):
        state = env.reset(key)
        if rollout_kind == "train":
            result, _ = rollout(env.step, action_fn, state, None, key, 2)
            return result.rewards.sum()
        if rollout_kind == "episode":
            result, _ = eval_rollout_episode(env.step, action_fn, state, None, key, 2)
            return result.rewards.sum()
        result, _ = fast_eval_rollout_episode(env.step, action_fn, state, None, key, 2)
        return result.episode_returns.sum()

    keys = jax.random.split(jax.random.key(3), 4)
    expected = jax.jit(instance)(keys[0])
    mesh = meta_mesh
    with jax.set_mesh(mesh):
        mapped = jax.shard_map(
            jax.vmap(instance),
            in_specs=jax.P(META_POP_AXIS_NAME),
            out_specs=jax.P(META_POP_AXIS_NAME),
        )
        inputs = jax.device_put(keys, jax.P(META_POP_AXIS_NAME))
        executable = jax.jit(mapped).lower(inputs).compile()
        hlo = executable.as_text()
        assert not compiled_collectives(hlo)
        actual = executable(inputs)
        np.testing.assert_allclose(np.asarray(actual)[0], expected, rtol=1e-6)


@pytest.mark.parametrize("target", ["paramppo", "paramopenes"])
def test_pbt_members_train_independently(target):
    root = Path(__file__).resolve().parents[2]
    common = OmegaConf.load(root / "configs/config.yaml")
    target_config = OmegaConf.merge(
        common,
        OmegaConf.load(root / f"configs/agent/meta/{target}.yaml"),
        {
            "num_envs": 2,
            "num_eval_envs": 2,
            "eval_episodes": 2,
            "normalize_obs": True,
            "random_timesteps": 4,
            "agent_network": {"actor_hidden_layer_sizes": [4]},
        },
    )
    if target == "paramppo":
        target_config.rollout_length = 2
        target_config.minibatch_size = 4
        target_config.reuse_rollout_epochs = 1
        target_config.agent_network.critic_hidden_layer_sizes = [4]
    else:
        target_config.num_envs = 1
        target_config.pop_size = 4
        target_config.episodes_for_fitness = 1
    config = OmegaConf.merge(
        common,
        OmegaConf.load(root / f"configs/agent/meta/pbt-{target}.yaml"),
        {
            "target_workflow": target_config,
            "pop_size": 4,
            "num_eval_envs": 2,
            "eval_episodes": 2,
            "workflow_steps_per_iter": 1,
            "warmup_steps": 100,
            "bottom_ratio": 0.25,
            "top_ratio": 0.25,
            "env": {"max_episode_steps": 2},
        },
    )
    target_cls = get_class(target_config.workflow_cls)
    original_target_step = target_cls.step
    # Both the PBT class and its dynamically loaded member class must be fresh:
    # build_from_config applies JIT transformations to class methods.
    member_cls = type(f"Member{target_cls.__name__}", (target_cls,), {})
    base = get_class(config.workflow_cls)
    cls = type(f"AxisProbe{base.__name__}", (base,), {})

    def create_env(*args, **kwargs):
        return ToyEnv(kwargs["parallel"])

    with (
        patch("hydra.utils.get_class", return_value=member_cls),
        patch(f"{target_cls.__module__}.create_env", side_effect=create_env),
        patch("evorl.algorithms.meta.pbt_workflow.create_env", side_effect=create_env),
    ):
        workflow = cls.build_from_config(config, jit=True)
    try:
        assert target_cls.step is original_target_step
        assert workflow.mesh.axis_names == (META_POP_AXIS_NAME,)
        assert META_POP_AXIS_NAME != POP_AXIS_NAME
        assert workflow.workflow.dp_axis_name is None
        assert not hasattr(workflow, "rollout_axis_name")
        assert not hasattr(workflow.workflow, "rollout_axis_name")
        with (
            patch("evorl.algorithms.ec.obs_utils.create_env", side_effect=create_env),
            jax.set_mesh(workflow.mesh),
        ):
            state = jax.jit(workflow.setup)(jax.random.key(0))
            if target == "paramopenes":
                # VMA inference must not reduce statistics over META_POP.
                np.testing.assert_array_equal(
                    state.pop_workflow_state.agent_state.obs_preprocessor_state.count,
                    [4] * config.pop_size,
                )

            @jax.shard_map(
                in_specs=jax.P(META_POP_AXIS_NAME),
                out_specs=jax.P(META_POP_AXIS_NAME),
            )
            def train_local_members(states):
                return jax.vmap(workflow.workflow.step)(states)

            executable = (
                jax.jit(train_local_members).lower(state.pop_workflow_state).compile()
            )
            hlo = executable.as_text()
            assert not compiled_collectives(hlo)
            _, trained = executable(state.pop_workflow_state)
            if target == "paramppo":
                # One population member must match its independent local update.
                member_index = jnp.array([0])
                member = jax.tree.map(
                    lambda x: x.at[member_index].get(out_sharding=jax.P())[0],
                    state.pop_workflow_state,
                )
                _, expected = workflow.workflow.step(member)
                for actual, reference in zip(
                    jax.tree.leaves(trained.agent_state.params),
                    jax.tree.leaves(expected.agent_state.params),
                    strict=True,
                ):
                    np.testing.assert_allclose(
                        actual.at[member_index].get(out_sharding=jax.P())[0],
                        reference,
                        rtol=1e-5,
                        atol=1e-6,
                    )
            _, state = workflow.step(state)
            metrics, state = workflow.evaluate(state)
            jax.block_until_ready(state)
            assert np.isfinite(metrics.pop_episode_returns).all()
            assert state.pop_workflow_state.key.sharding.spec == jax.P(
                META_POP_AXIS_NAME
            )
    finally:
        workflow.close()
