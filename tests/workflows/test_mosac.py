"""MOSAC replay ownership, population updates and complete functional training."""

from pathlib import Path
from unittest.mock import Mock, patch

from flax import nnx
from hydra import compose, initialize_config_dir
from hydra.utils import get_class
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.algorithms.multi_objective.mosac import MOSACWorkflow
from evorl.envs import AutoresetMode
from evorl.envs.wrappers import (
    ActionSquashWrapper,
    EpisodeWrapper,
    OneEpisodeWrapper,
    VmapAutoResetWrapper,
    VmapWrapper,
)
from evorl.utils.orbax_utils import load, save
from tests.helpers.multi_objective_envs import ToyMultiObjectiveEnv

pytestmark = pytest.mark.integration


def make_mosac(
    continuous=True, jit=True, multi_device=False, adaptive_alpha=True, overrides=None
):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config", overrides=["agent=multi_objective/mosac"]
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            num_envs=4,
            num_eval_envs=4,
            eval_episodes=4,
            rollout_length=2,
            random_timesteps=8,
            learning_start_timesteps=12,
            batch_size=4,
            replay_buffer_capacity=32,
            num_updates_per_iter=2,
            fold_iters=2,
            total_timesteps=28,
            normalize_obs=True,
            adaptive_alpha=adaptive_alpha,
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
        overrides or {},
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

    cls = type("TestMOSAC", (MOSACWorkflow,), {})
    with patch(
        "evorl.algorithms.multi_objective.mosac.create_env", side_effect=create_env
    ):
        return cls.build_from_config(config, jit=jit, multi_device=multi_device)


def assert_tree_equal(first, second):
    for x, y in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("continuous", [True, False])
@pytest.mark.parametrize("adaptive_alpha", [True, False])
def test_mosac_warmup_step_and_replay_only_update(continuous, adaptive_alpha, jit):
    workflow = make_mosac(continuous, jit=jit, adaptive_alpha=adaptive_alpha)
    try:
        state = workflow.setup(jax.random.key(0))
        assert int(state.metrics.sampled_timesteps) == 12
        assert int(state.metrics.sampled_episodes) == 0
        # Keep warmup transitions, but start training from a fresh episode.
        np.testing.assert_array_equal(state.env_state.env_state, 0)
        np.testing.assert_array_equal(
            state.replay_buffer_state.data.extras.env_extras.ori_obs[4:8, 0], 2
        )
        np.testing.assert_array_equal(
            state.replay_buffer_state.data.extras.env_extras.termination[4:8], 1
        )
        original = jax.tree.map(jnp.copy, state)
        train, updated = workflow.update(state)
        assert np.isfinite(train.loss)
        assert int(updated.metrics.sampled_timesteps) == 12
        assert int(updated.metrics.iterations) == 0
        assert int(updated.metrics.training_updates) == 4
        assert_tree_equal(state.env_state, updated.env_state)
        assert_tree_equal(state.replay_buffer_state, updated.replay_buffer_state)
        assert_tree_equal(original, state)
        train, updated = workflow.step(updated)
        assert np.isfinite(train.loss)
        assert int(updated.metrics.sampled_timesteps) == 20
        assert int(updated.metrics.sampled_episodes) == 4
        assert int(updated.metrics.training_updates) == 8
        evaluation, _ = workflow.evaluate(updated)
        assert evaluation.episode_return.shape == (2,)
        assert evaluation.episode_length.shape == ()
        np.testing.assert_allclose(
            evaluation.scalarized_return,
            evaluation.preferences @ evaluation.episode_return,
        )
    finally:
        workflow.close()


def test_mosac_nested_vmap_and_one_read_only_shared_replay():
    workflow = make_mosac()
    try:
        keys = jax.random.split(jax.random.key(1), 4).reshape(2, 2)
        state = jax.jit(jax.vmap(jax.vmap(workflow.setup)))(keys)
        metrics, state = jax.jit(jax.vmap(jax.vmap(workflow.step)))(state)
        assert metrics.loss.shape == (2, 2)
        shared = jax.tree.map(lambda x: x[0, 0], state.replay_buffer_state)
        state = state.replace(replay_buffer_state=None)

        def update(member):
            metric, member = workflow.update(member.replace(replay_buffer_state=shared))
            return metric, member.replace(replay_buffer_state=None)

        metrics, updated = jax.jit(jax.vmap(jax.vmap(update)))(state)
        assert updated.replay_buffer_state is None
        assert metrics.loss.shape == (2, 2)
        np.testing.assert_array_equal(
            updated.metrics.sampled_timesteps, state.metrics.sampled_timesteps
        )
        np.testing.assert_array_equal(
            updated.metrics.training_updates, state.metrics.training_updates + 4
        )
    finally:
        workflow.close()


def test_mosac_complete_learn_and_checkpoint_resume(tmp_path):
    workflow = make_mosac()
    workflow.recorder, workflow.checkpoint_manager = Mock(), Mock()
    try:
        state = workflow.learn(workflow.setup(jax.random.key(2)))
        assert int(state.metrics.sampled_timesteps) == 28
        assert int(state.metrics.training_updates) == 8
        assert workflow.recorder.write.called
        assert workflow.checkpoint_manager.save.call_args.kwargs["force"]
        save(tmp_path / "state", state)
        restored = load(tmp_path / "state", state)
        expected_metric, expected = workflow.update(state)
        actual_metric, actual = workflow.update(restored)
        assert_tree_equal(expected_metric, actual_metric)
        assert_tree_equal(expected, actual)
        workflow.recorder.reset_mock()
        workflow.learn(restored)
        workflow.recorder.write.assert_not_called()
    finally:
        workflow.close()


@pytest.mark.parametrize("continuous", [True, False])
@pytest.mark.parametrize("adaptive_alpha", [True, False])
def test_mosac_two_device_step_and_standalone_update(
    continuous, adaptive_alpha, mesh_factory
):
    mesh_factory()
    workflow = make_mosac(continuous, multi_device=True, adaptive_alpha=adaptive_alpha)
    try:
        with jax.set_mesh(workflow.mesh):
            state = workflow.setup(jax.random.key(3))
            train, state = workflow.step(state)
            pure_update, state = workflow.update(state)
            evaluation, _ = workflow.evaluate(state)
        assert np.isfinite(train.loss)
        assert np.isfinite(pure_update.loss)
        assert int(state.metrics.sampled_timesteps) == 20
        assert int(state.metrics.training_updates) == 8
        assert evaluation.episode_return.shape == (2,)
    finally:
        workflow.close()


@pytest.mark.parametrize(
    "example", ["mosac/playground-cartpole", "mosac/deep-sea-treasure"]
)
def test_mosac_real_native_training(example):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config", overrides=[f"agent=exp/multi_objective/{example}"]
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            num_envs=2,
            num_eval_envs=2,
            eval_episodes=2,
            rollout_length=1,
            random_timesteps=2,
            learning_start_timesteps=2,
            batch_size=2,
            replay_buffer_capacity=16,
            num_updates_per_iter=2,
            fold_iters=1,
            total_timesteps=4,
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
    )
    cls = type(f"Real{example}", (get_class(config.workflow_cls),), {})
    workflow = cls.build_from_config(config)
    try:
        state = workflow.setup(jax.random.key(12))
        train, state = workflow.step(state)
        evaluation, _ = workflow.evaluate(state)
        assert np.isfinite(train.loss)
        assert np.isfinite(evaluation.episode_return).all()
        assert int(state.metrics.sampled_timesteps) == 4
        assert evaluation.episode_return.shape == (2,)
    finally:
        workflow.close()


@pytest.mark.parametrize(
    "example", ["mosac/mo-mountaincar", "mosac/mo-deep-sea-treasure"]
)
def test_mosac_host_nested_vmap_training(example, mo_gymnasium_pools):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config", overrides=[f"agent=exp/multi_objective/{example}"]
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            num_envs=2,
            num_eval_envs=2,
            eval_episodes=2,
            rollout_length=1,
            random_timesteps=2,
            learning_start_timesteps=2,
            batch_size=2,
            replay_buffer_capacity=16,
            num_updates_per_iter=2,
            fold_iters=1,
            total_timesteps=4,
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
    )
    cls = type(f"Host{example}", (get_class(config.workflow_cls),), {})
    workflow = cls.build_from_config(config)
    try:
        keys = jax.random.split(jax.random.key(13), 4).reshape(2, 2)
        state = jax.jit(jax.vmap(jax.vmap(workflow.setup)))(keys)
        train, state = jax.jit(jax.vmap(jax.vmap(workflow.step)))(state)
        evaluation, _ = jax.jit(jax.vmap(jax.vmap(workflow.evaluate)))(state)
        jax.effects_barrier()
        assert np.isfinite(train.loss).all()
        assert np.isfinite(evaluation.episode_return).all()
        assert evaluation.episode_return.shape == (2, 2, 2)
        np.testing.assert_array_equal(state.metrics.sampled_timesteps, 4)
    finally:
        workflow.close()


@pytest.mark.parametrize("random_steps,policy_start,expected", [(0, 8, 8), (8, 4, 8)])
def test_mosac_empty_warmup_phases(random_steps, policy_start, expected):
    workflow = make_mosac(
        overrides=dict(
            random_timesteps=random_steps, learning_start_timesteps=policy_start
        )
    )
    try:
        state = workflow.setup(jax.random.key(15))
        assert int(state.metrics.sampled_timesteps) == expected
        assert int(state.replay_buffer_state.buffer_size) == expected
        train, _ = workflow.update(state)
        assert np.isfinite(train.loss)
    finally:
        workflow.close()


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("interval", [1, 2, 3])
def test_mosac_actor_interval_and_target_update_match_sac(continuous, interval):
    workflow = make_mosac(
        continuous=continuous,
        adaptive_alpha=True,
        overrides=dict(num_updates_per_iter=1, actor_update_interval=interval, tau=0.2),
    )
    try:
        state = workflow.setup(jax.random.key(16))
        original = workflow.agent.get_models(state.agent_state, copy=True)
        target = nnx.state(original.target_critic, nnx.Param)
        for cycle in range(1, 4):
            _, state = workflow.update(state)
            models = workflow.agent.get_models(state.agent_state, copy=True)
            assert int(nnx.merge(state.opt_state.actor, copy=True).step[...]) == cycle
            assert int(nnx.merge(state.opt_state.alpha, copy=True).step[...]) == cycle
            assert (
                int(nnx.merge(state.opt_state.critic, copy=True).step[...])
                == cycle * interval
            )
            assert int(state.metrics.training_updates) == cycle * interval
            current = nnx.state(models.critic, nnx.Param)
            target = jax.tree.map(
                lambda old, new: 0.8 * old + 0.2 * new, target, current
            )
            assert_tree_equal(target, nnx.state(models.target_critic, nnx.Param))
        assert int(state.metrics.sampled_timesteps) == 12
    finally:
        workflow.close()
