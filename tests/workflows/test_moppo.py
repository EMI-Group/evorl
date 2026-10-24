"""End-to-end MOPPO updates, parallel instances and resumable training."""

from pathlib import Path
from unittest.mock import Mock, patch

from hydra import compose, initialize_config_dir
from hydra.utils import get_class
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.algorithms.multi_objective.moppo import MOPPOWorkflow
from evorl.envs import AutoresetMode
from evorl.envs.wrappers import (
    ActionSquashWrapper,
    EpisodeWrapper,
    OneEpisodeWrapper,
    VmapEnvPoolAutoResetWrapper,
    VmapWrapper,
)
from evorl.rollout import rollout
from evorl.utils.orbax_utils import load, save
from evorl.utils.rl_toolkits import compute_gae
from tests.helpers.multi_objective_envs import ToyMultiObjectiveEnv

pytestmark = pytest.mark.integration


def make_moppo(jit=True, multi_device=False, continuous=True, episode_length=3):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config", overrides=["agent=multi_objective/moppo"]
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=episode_length),
            num_envs=4,
            num_eval_envs=4,
            eval_episodes=4,
            rollout_length=2,
            minibatch_size=4,
            reuse_rollout_epochs=2,
            total_timesteps=16,
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
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
        assert autoreset_mode == AutoresetMode.ENVPOOL
        return VmapEnvPoolAutoResetWrapper(
            EpisodeWrapper(env, config.max_episode_steps), parallel
        )

    cls = type("TestMOPPO", (MOPPOWorkflow,), {})
    with patch(
        "evorl.algorithms.multi_objective.moppo.create_env", side_effect=create_env
    ):
        return cls.build_from_config(config, jit=jit, multi_device=multi_device)


@pytest.mark.parametrize("episode_length", [1, 3])
def test_envpool_gae_uses_final_observation_at_episode_boundaries(episode_length):
    workflow = make_moppo(continuous=False, episode_length=episode_length)
    try:
        state = workflow.setup(jax.random.key(17))
        trajectory, _ = jax.jit(
            lambda env_state: rollout(
                workflow.env.step,
                workflow.agent.compute_actions,
                env_state,
                state.agent_state,
                jax.random.key(18),
                rollout_length=3,
                env_extra_fields=("autoreset", "termination"),
            )
        )(state.env_state)
        observations = jnp.concatenate(
            [trajectory.obs, trajectory.next_obs[-1:]], axis=0
        )
        values = workflow.agent.compute_values(
            state.agent_state, trajectory.replace(obs=observations)
        )
        targets, _ = compute_gae(
            trajectory.rewards,
            values,
            trajectory.dones[..., None],
            trajectory.extras.env_extras.termination[..., None],
            discount=workflow.config.discount,
            gae_lambda=workflow.config.gae_lambda,
        )
        boundary = np.asarray(trajectory.dones, bool)
        # ENVPOOL exposes the final observation before its reset-only step.
        np.testing.assert_array_equal(
            np.asarray(trajectory.next_obs)[boundary, 0], min(episode_length, 2)
        )
        assert bool(trajectory.extras.env_extras.autoreset.any())
        expected = (
            trajectory.rewards
            + workflow.config.discount
            * (1 - trajectory.extras.env_extras.termination[..., None])
            * values[1:]
        )
        np.testing.assert_allclose(targets[boundary], expected[boundary], atol=1e-6)
        # The short episode case also truncates on the rollout's last step.
        if episode_length == 1:
            assert bool(trajectory.dones[-1].all())
            assert not bool(trajectory.extras.env_extras.termination.any())
        else:
            assert bool(trajectory.extras.env_extras.termination.any())
    finally:
        workflow.close()


@pytest.mark.parametrize("continuous", [True, False])
def test_moppo_step_and_vector_evaluation(continuous, jit):
    workflow = make_moppo(jit=jit, continuous=continuous)
    try:
        state = workflow.setup(jax.random.key(0))
        previous = jax.tree.map(jnp.copy, state)
        metrics, updated = workflow.step(state)
        assert np.isfinite(metrics.loss)
        assert int(updated.metrics.sampled_timesteps) == 8
        assert int(updated.metrics.sampled_episodes) == 4
        assert int(updated.metrics.training_updates) == 4
        evaluation, _ = workflow.evaluate(updated)
        assert evaluation.episode_returns.shape == (2,)
        assert evaluation.episode_lengths.shape == ()
        np.testing.assert_allclose(
            evaluation.scalarized_returns,
            evaluation.episode_returns @ evaluation.preferences,
        )
        assert any(
            not np.array_equal(x, y)
            for x, y in zip(
                jax.tree.leaves(state.agent_state.model_state),
                jax.tree.leaves(updated.agent_state.model_state),
            )
        )
        for x, y in zip(jax.tree.leaves(previous), jax.tree.leaves(state)):
            np.testing.assert_array_equal(
                jax.random.key_data(x)
                if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
                else x,
                jax.random.key_data(y)
                if jax.dtypes.issubdtype(y.dtype, jax.dtypes.prng_key)
                else y,
            )
    finally:
        workflow.close()


def test_moppo_nested_vmap():
    workflow = make_moppo()
    try:
        keys = jax.random.split(jax.random.key(2), 4).reshape(2, 2)
        state = jax.jit(jax.vmap(jax.vmap(workflow.setup)))(keys)
        metrics, state = jax.jit(jax.vmap(jax.vmap(workflow.step)))(state)
        evaluation, _ = jax.jit(jax.vmap(jax.vmap(workflow.evaluate)))(state)
        assert metrics.loss.shape == (2, 2)
        assert evaluation.episode_returns.shape == (2, 2, 2)
        assert bool(jnp.isfinite(metrics.loss).all())
        metrics, _ = jax.jit(jax.vmap(jax.vmap(workflow.step)))(state)
        assert bool(jnp.isfinite(metrics.loss).all())
    finally:
        workflow.close()


def test_moppo_learn_checkpoint_and_completed_budget(tmp_path):
    workflow = make_moppo()
    workflow.recorder = Mock()
    workflow.checkpoint_manager = Mock()
    try:
        state = workflow.learn(workflow.setup(jax.random.key(3)))
        assert int(state.metrics.sampled_timesteps) == 16
        assert workflow.recorder.write.call_count >= 4
        path = tmp_path / "state"
        save(path, state)
        restored = load(path, state)
        _, expected = workflow.step(state)
        _, actual = workflow.step(restored)
        for x, y in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
            np.testing.assert_allclose(
                jax.random.key_data(x)
                if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
                else x,
                jax.random.key_data(y)
                if jax.dtypes.issubdtype(y.dtype, jax.dtypes.prng_key)
                else y,
            )
        workflow.recorder.reset_mock()
        state = workflow.learn(state)
        workflow.recorder.write.assert_not_called()
    finally:
        workflow.close()


def test_moppo_two_device_step(mesh_factory):
    mesh_factory()
    workflow = make_moppo(multi_device=True)
    try:
        with jax.set_mesh(workflow.mesh):
            state = workflow.setup(jax.random.key(4))
            metrics, state = workflow.step(state)
            assert int(state.metrics.sampled_timesteps) == 8
            metrics, state = workflow.step(state)
            evaluation, _ = workflow.evaluate(state)
        assert np.isfinite(metrics.loss)
        assert int(state.metrics.sampled_timesteps) == 16
        assert evaluation.episode_returns.shape == (2,)
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["moppo", "pgmorl"])
def test_moppo_family_real_playground_training(algorithm):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config",
            overrides=[f"agent=exp/multi_objective/{algorithm}/playground-cartpole"],
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            num_envs=2,
            num_eval_envs=2,
            eval_episodes=2,
            rollout_length=2,
            minibatch_size=4,
            reuse_rollout_epochs=1,
            normalize_obs=True,
            total_timesteps=8,
            pop_size=2,
            warmup_iterations=1,
            evolutionary_iterations=1,
            performance_buffer_divisions=2,
            performance_buffer_size=1,
            weight_divisions=2,
            archive_capacity=4,
            reference_point=[0.0, 0.0],
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
    )

    cls = type(f"Real{algorithm}", (get_class(config.workflow_cls),), {})
    workflow = cls.build_from_config(config)
    try:
        state = workflow.setup(jax.random.key(7))
        metric, state = workflow.step(state)
        evaluation, _ = workflow.evaluate(state)
        assert np.isfinite(metric.loss)
        assert np.isfinite(evaluation.episode_returns).all()
        assert np.all(evaluation.episode_lengths > 0)
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["moppo", "pgmorl"])
def test_moppo_family_host_multi_objective_training(algorithm, mo_gymnasium_pools):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config",
            overrides=[f"agent=exp/multi_objective/{algorithm}/mo-mountaincar"],
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            num_envs=2,
            num_eval_envs=2,
            eval_episodes=2,
            rollout_length=2,
            minibatch_size=4,
            reuse_rollout_epochs=1,
            total_timesteps=4,
            pop_size=2,
            warmup_iterations=1,
            evolutionary_iterations=1,
            performance_buffer_divisions=2,
            performance_buffer_size=1,
            weight_divisions=2,
            archive_capacity=4,
            reference_point=[-20.0, -20.0],
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
    )
    cls = type(f"Host{algorithm}", (get_class(config.workflow_cls),), {})
    workflow = cls.build_from_config(config)
    try:
        if algorithm == "moppo":
            keys = jax.random.split(jax.random.key(8), 4).reshape(2, 2)
            state = jax.jit(jax.vmap(jax.vmap(workflow.setup)))(keys)
            metric, state = jax.jit(jax.vmap(jax.vmap(workflow.step)))(state)
            evaluation, _ = jax.jit(jax.vmap(jax.vmap(workflow.evaluate)))(state)
            assert evaluation.episode_returns.shape == (2, 2, 2)
        else:
            state = workflow.setup(jax.random.key(8))
            metric, state = workflow.step(state)
            # The native pool advances both population members together even
            # when only one worker's rollout fits the remaining sample budget.
            assert int(state.metrics.sampled_timesteps) == 8
            before = int(state.metrics.iterations)
            state = workflow.learn(state)
            assert int(state.metrics.iterations) == before
            evaluation, _ = workflow.evaluate(state)
            assert evaluation.episode_returns.shape[1] == 2
        jax.effects_barrier()
        assert np.isfinite(metric.loss).all()
        assert np.isfinite(evaluation.episode_returns).all()
        assert np.all(evaluation.episode_lengths == 3)
    finally:
        workflow.close()
