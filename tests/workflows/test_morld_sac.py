"""MORL/D-SAC replay ownership, independent learners and resumable training."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from flax import nnx
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.algorithms.multi_objective.morld_sac import MORLDSACWorkflow
from evorl.envs import AutoresetMode
from evorl.envs.wrappers import (
    ActionSquashWrapper,
    EpisodeWrapper,
    OneEpisodeWrapper,
    VmapAutoResetWrapper,
    VmapWrapper,
)
from evorl.replay_buffers import ReplayBufferState
from evorl.utils.orbax_utils import load, save
from tests.helpers.multi_objective_envs import ToyMultiObjectiveEnv

pytestmark = pytest.mark.integration


def make_morld_sac(
    jit=True,
    shared=True,
    continuous=True,
    transfer=False,
    passes=2,
    pop_size=2,
    multi_device=False,
):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config", overrides=["agent=multi_objective/morld_sac"]
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            pop_size=pop_size,
            archive_capacity=4,
            total_timesteps=24,
            exchange_every=4,
            shared_replay_buffer=shared,
            transfer=transfer,
            update_passes=passes,
            target_workflow=dict(
                num_envs=2,
                num_eval_envs=2,
                eval_episodes=2,
                rollout_length=2,
                batch_size=4,
                replay_buffer_capacity=16,
                random_timesteps=8,
                learning_start_timesteps=8,
                num_updates_per_iter=2,
                actor_update_interval=1,
                adaptive_alpha=True,
                normalize_obs=True,
                agent_network=dict(
                    actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
                ),
            ),
        ),
    )

    def create_env(config, parallel, autoreset_mode, **kwargs):
        env = ToyMultiObjectiveEnv(continuous)
        if continuous:
            env = ActionSquashWrapper(env)
        if autoreset_mode == AutoresetMode.DISABLED:
            return VmapWrapper(
                OneEpisodeWrapper(env, config.max_episode_steps),
                parallel,
                vmap_step=True,
            )
        return VmapAutoResetWrapper(
            EpisodeWrapper(env, config.max_episode_steps), parallel
        )

    cls = type("TestMORLDSAC", (MORLDSACWorkflow,), {})
    with patch(
        "evorl.algorithms.multi_objective.mosac.create_env", side_effect=create_env
    ):
        return cls.build_from_config(config, jit=jit, multi_device=multi_device)


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("continuous", [True, False])
def test_morld_sac_data_parallel_local_replay_and_independent_policies(
    shared, continuous
):
    workflow = make_morld_sac(
        multi_device=True, shared=shared, continuous=continuous, transfer=True
    )
    try:
        with jax.set_mesh(workflow.mesh):
            state = workflow.setup(jax.random.key(37))
            warmup = 8 if shared else 16
            assert int(state.metrics.sampled_timesteps) == warmup
            executable = jax.jit(workflow.step).lower(state).compile()
            assert "all-gather" not in executable.as_text().lower()
            metric, updated = executable(state)
            assert np.isfinite(metric.loss)
            assert int(updated.metrics.sampled_timesteps) == warmup + 4
            assert int(updated.metrics.training_updates) == 6
            assert int(updated.current_policy) == 1
            replay = (
                updated.replay_buffer_state
                if shared
                else updated.population.replay_buffer_state
            )
            if workflow.num_devices > 1:
                assert not replay.data.obs.sharding.is_fully_replicated
            assert replay.data.obs.shape[:2] == ((16, 3) if shared else (2, 16))
            for leaf in jax.tree.leaves(updated.population.agent_state.model_state):
                assert leaf.sharding.is_fully_replicated
            evaluation, _ = workflow.evaluate(updated)
            assert np.isfinite(evaluation.episode_returns).all()
    finally:
        workflow.close()


def assert_tree_close(expected, actual, *, num_devices=None):
    def visible_replay(value):
        if not isinstance(value, ReplayBufferState):
            return value
        sizes = np.asarray(value.buffer_size)
        # Singleton sharding annotations can disappear after slicing a vmapped
        # result. Use the workflow's execution layout to mask unwritten slots.
        distributed = num_devices is not None
        leading = sizes.ndim - int(distributed)
        capacity = jax.tree.leaves(value.data)[0].shape[leading]
        if distributed:
            local_capacity = capacity // num_devices
            valid = np.arange(capacity) % local_capacity < np.repeat(
                sizes, local_capacity, axis=-1
            )
        else:
            valid = np.arange(capacity) < sizes[..., None]
        data = jax.tree.map(
            lambda x: np.where(
                valid.reshape((*valid.shape, *((1,) * (x.ndim - valid.ndim)))),
                np.asarray(x),
                0,
            ),
            value.data,
        )
        return value.replace(data=data)

    # Unwritten jnp.empty storage is not part of the logical replay state.
    is_buffer = lambda x: isinstance(x, ReplayBufferState)
    expected = jax.tree.map(visible_replay, expected, is_leaf=is_buffer)
    actual = jax.tree.map(visible_replay, actual, is_leaf=is_buffer)
    for x, y in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        np.testing.assert_allclose(x, y, atol=1e-6)


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("continuous", [True, False])
def test_replay_ownership_and_training_counts(shared, continuous, jit):
    workflow = make_morld_sac(jit=jit, shared=shared, continuous=continuous)
    try:
        state = workflow.setup(jax.random.key(0))
        np.testing.assert_array_equal(state.population.env_state.env_state, 0)
        assert int(state.metrics.sampled_timesteps) == (8 if shared else 16)
        assert int(state.metrics.sampled_episodes) == (4 if shared else 8)
        if shared:
            assert state.population.replay_buffer_state is None
            assert state.replay_buffer_state.data.rewards.shape == (16, 2)
            assert int(state.replay_buffer_state.buffer_size) == 8
        else:
            assert state.replay_buffer_state is None
            assert state.population.replay_buffer_state.data.rewards.shape == (2, 16, 2)
            np.testing.assert_array_equal(
                state.population.replay_buffer_state.buffer_size, [8, 8]
            )
        train, updated = workflow.step(state)
        assert np.isfinite(train.loss)
        assert (
            int(updated.metrics.sampled_timesteps - state.metrics.sampled_timesteps)
            == 4
        )
        assert int(updated.metrics.training_updates) == 6
        np.testing.assert_array_equal(
            updated.population.metrics.training_updates, [2, 4]
        )
        np.testing.assert_array_equal(updated.population.metrics.iterations, [1, 0])
        if shared:
            assert updated.population.replay_buffer_state is None
            assert int(updated.replay_buffer_state.buffer_size) == 12
            ring = updated.replay_buffer_state
        else:
            np.testing.assert_array_equal(
                updated.population.replay_buffer_state.buffer_size, [12, 8]
            )
            old = jax.tree.map(lambda x: x[1], state.population.replay_buffer_state)
            actual = jax.tree.map(
                lambda x: x[1], updated.population.replay_buffer_state
            )
            assert_tree_close(old, actual)
            ring = jax.tree.map(lambda x: x[0], updated.population.replay_buffer_state)
        # Only collection writes replay; other learners' replay-only updates
        # cannot append the same four transitions again or collapse reward axes.
        assert int(ring.current_index) == 12
        np.testing.assert_array_equal(
            ring.data.extras.env_extras.ori_obs[8:12, 0], [1, 1, 2, 2]
        )
        np.testing.assert_array_equal(
            ring.data.extras.env_extras.termination[8:12], [0, 0, 1, 1]
        )
        evaluation, _ = workflow.evaluate(updated)
        assert evaluation.episode_returns.shape == (2, 2)
        assert np.isfinite(evaluation.episode_returns).all()
        assert int(state.metrics.training_updates) == 0
    finally:
        workflow.close()


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("multi_device", [False, True])
def test_nested_instances_keep_independent_keys_and_replay(shared, multi_device):
    workflow = make_morld_sac(shared=shared, multi_device=multi_device)
    try:
        with jax.set_mesh(workflow.mesh) if multi_device else nullcontext():
            keys = jax.random.split(jax.random.key(1), 4).reshape(2, 2)
            state = jax.jit(jax.vmap(jax.vmap(workflow.setup)))(keys)
            train, updated = jax.jit(jax.vmap(jax.vmap(workflow.step)))(state)
            assert train.loss.shape == (2, 2)
            assert bool(jnp.isfinite(train.loss).all())
            for i in range(2):
                for j in range(2):
                    independent = workflow.setup(keys[i, j])
                    _, independent = workflow.step(independent)
                    batched = jax.tree.map(lambda x, i=i, j=j: x[i, j], updated)
                    assert_tree_close(
                        independent,
                        batched,
                        num_devices=workflow.num_devices if multi_device else None,
                    )
    finally:
        workflow.close()


@pytest.mark.parametrize("shared", [True, False])
def test_multiple_other_learners_update_without_duplicate_replay_writes(shared):
    workflow = make_morld_sac(shared=shared, pop_size=3)
    try:
        state = workflow.setup(jax.random.key(18))
        _, updated = workflow.step(state)
        np.testing.assert_array_equal(
            updated.population.metrics.training_updates, [2, 4, 4]
        )
        np.testing.assert_array_equal(updated.population.metrics.iterations, [1, 0, 0])
        assert int(updated.metrics.training_updates) == 10
        assert (
            int(updated.metrics.sampled_timesteps - state.metrics.sampled_timesteps)
            == 4
        )
        if shared:
            assert updated.population.replay_buffer_state is None
            assert int(updated.replay_buffer_state.buffer_size) == 12
        else:
            np.testing.assert_array_equal(
                updated.population.replay_buffer_state.buffer_size, [12, 8, 8]
            )
        for i in (1, 2):
            previous = jax.tree.map(lambda x, i=i: x[i], state.population)
            member = jax.tree.map(lambda x, i=i: x[i], updated.population)
            assert_tree_close(previous.env_state, member.env_state)
            assert_tree_close(previous.replay_buffer_state, member.replay_buffer_state)
            assert any(
                not np.array_equal(x, y)
                for x, y in zip(
                    jax.tree.leaves(previous.agent_state.model_state),
                    jax.tree.leaves(member.agent_state.model_state),
                    strict=True,
                )
            )
        keys = jax.random.key_data(updated.population.key)
        assert not np.array_equal(keys[1], keys[2])
    finally:
        workflow.close()


@pytest.mark.parametrize("shared", [True, False])
def test_actor_transfer_preserves_recipient_critic_temperature_and_replay(shared):
    workflow = make_morld_sac(shared=shared, transfer=True, passes=0)
    try:
        state = workflow.setup(jax.random.key(2))
        target_before = jax.tree.map(lambda x: x[1], state.population)
        _, updated = workflow.step(state)
        source = jax.tree.map(lambda x: x[0], updated.population)
        target = jax.tree.map(lambda x: x[1], updated.population)
        agent = workflow.workflow.agent
        a = agent.get_models(source.agent_state)
        b = agent.get_models(target.agent_state)
        old = agent.get_models(target_before.agent_state)
        assert_tree_close(nnx.state(a.actor), nnx.state(b.actor))
        assert_tree_close(nnx.state(old.critic), nnx.state(b.critic))
        assert_tree_close(nnx.state(old.target_critic), nnx.state(b.target_critic))
        assert_tree_close(target_before.opt_state.critic, target.opt_state.critic)
        assert_tree_close(target_before.opt_state.alpha, target.opt_state.alpha)
        assert_tree_close(target_before.replay_buffer_state, target.replay_buffer_state)
        assert_tree_close(target_before.env_state, target.env_state)
        assert_tree_close(
            target_before.agent_state.extra_state, target.agent_state.extra_state
        )
    finally:
        workflow.close()


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("multi_device", [False, True])
def test_learn_checkpoint_preserves_ring_pointer_and_future_updates(
    shared, tmp_path, multi_device, mesh_factory
):
    if multi_device:
        mesh_factory()
    workflow = make_morld_sac(shared=shared, multi_device=multi_device)
    workflow.recorder, workflow.checkpoint_manager = Mock(), Mock()
    try:
        with jax.set_mesh(workflow.mesh) if multi_device else nullcontext():
            state = workflow.learn(workflow.setup(jax.random.key(3)))
            assert int(state.metrics.sampled_timesteps) == 24
            assert workflow.checkpoint_manager.save.call_args.kwargs["force"]
            path = tmp_path / "morld_sac"
            save(path, state)
            restored = load(path, state)
            _, expected = workflow.step(state)
            _, actual = workflow.step(restored)
            assert_tree_close(
                expected,
                actual,
                num_devices=workflow.num_devices if multi_device else None,
            )
            workflow.recorder.reset_mock()
            workflow.learn(restored)
            workflow.recorder.write.assert_not_called()
    finally:
        workflow.close()


@pytest.mark.parametrize(
    "environment, multi_device",
    [
        ("deep-sea-treasure", False),
        ("playground-cartpole", False),
        ("mo-mountaincar", False),
        ("deep-sea-treasure", True),
        ("playground-cartpole", True),
    ],
)
def test_real_morld_sac_training(environment, request, multi_device):
    if environment == "mo-mountaincar":
        request.getfixturevalue("mo_gymnasium_pools")
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config",
            overrides=[f"agent=exp/multi_objective/morld_sac/{environment}"],
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            pop_size=2,
            archive_capacity=4,
            exchange_every=2,
            total_timesteps=6,
            update_passes=1,
            target_workflow=dict(
                num_envs=2,
                num_eval_envs=2,
                eval_episodes=2,
                rollout_length=1,
                batch_size=4,
                random_timesteps=4,
                learning_start_timesteps=4,
                replay_buffer_capacity=16,
                num_updates_per_iter=1,
                agent_network=dict(
                    actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
                ),
            ),
        ),
    )
    cls = type("RealMORLDSAC", (MORLDSACWorkflow,), {})
    workflow = cls.build_from_config(config, multi_device=multi_device)
    workflow.recorder, workflow.checkpoint_manager = Mock(), Mock()
    try:
        with jax.set_mesh(workflow.mesh) if multi_device else nullcontext():
            try:
                initial = workflow.setup(jax.random.key(4))
            except TypeError as error:
                if (
                    multi_device
                    and environment == "playground-cartpole"
                    and "[0].swap" in str(error)
                    and "manual axis types do not match" in str(error)
                ):
                    pytest.xfail(
                        "Upstream MJX solver._while_loop_scan has inconsistent VMA types"
                    )
                raise
            state = workflow.learn(initial)
            assert int(state.metrics.sampled_timesteps) == 6
            assert state.population.replay_buffer_state is None
            np.testing.assert_array_equal(
                state.replay_buffer_state.buffer_size,
                [6 // workflow.num_devices] * workflow.num_devices
                if multi_device
                else 6,
            )
            evaluation, _ = workflow.evaluate(state)
            assert np.isfinite(evaluation.episode_returns).all()
            assert np.all(np.asarray(evaluation.episode_lengths) > 0)
    finally:
        workflow.close()


def test_nested_host_shared_replay_uses_the_selected_member_batch(mo_gymnasium_pools):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config",
            overrides=["agent=exp/multi_objective/morld_sac/mo-mountaincar"],
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            pop_size=3,
            archive_capacity=4,
            exchange_every=2,
            total_timesteps=8,
            update_passes=1,
            target_workflow=dict(
                num_envs=2,
                num_eval_envs=2,
                eval_episodes=2,
                rollout_length=1,
                batch_size=4,
                random_timesteps=4,
                learning_start_timesteps=4,
                replay_buffer_capacity=16,
                num_updates_per_iter=1,
                agent_network=dict(
                    actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
                ),
            ),
        ),
    )
    cls = type("NestedHostMORLDSAC", (MORLDSACWorkflow,), {})
    workflow = cls.build_from_config(config)
    try:
        keys = jax.random.split(jax.random.key(21), 4).reshape(2, 2)
        state = jax.jit(jax.vmap(jax.vmap(workflow.setup)))(keys)
        train, state = jax.jit(jax.vmap(jax.vmap(workflow.step)))(state)
        evaluation, _ = jax.jit(jax.vmap(jax.vmap(workflow.evaluate)))(state)
        assert train.loss.shape == (2, 2)
        assert evaluation.episode_returns.shape == (2, 2, 3, 2)
        assert state.population.replay_buffer_state is None
        assert state.replay_buffer_state.data.rewards.shape == (2, 2, 16, 2)
        np.testing.assert_array_equal(
            state.replay_buffer_state.buffer_size, np.full((2, 2), 6)
        )
        np.testing.assert_array_equal(
            state.metrics.sampled_timesteps, np.full((2, 2), 6)
        )
        np.testing.assert_array_equal(
            state.metrics.training_updates, np.full((2, 2), 6)
        )
        assert np.isfinite(train.loss).all()
        assert np.isfinite(evaluation.episode_returns).all()
        jax.effects_barrier()
    finally:
        workflow.close()
