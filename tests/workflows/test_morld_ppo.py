"""MORL/D independent policies, actor transfer and resumable population training."""

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

from evorl.algorithms.multi_objective.morld_ppo import MORLDPPOWorkflow
from evorl.envs import AutoresetMode
from evorl.envs.wrappers import (
    ActionSquashWrapper,
    EpisodeWrapper,
    OneEpisodeWrapper,
    VmapEnvPoolAutoResetWrapper,
    VmapWrapper,
)
from evorl.utils.orbax_utils import load, save
from tests.helpers.multi_objective_envs import ToyMultiObjectiveEnv

pytestmark = pytest.mark.integration


def make_morld(
    jit=True, objectives=2, transfer=True, adaptation=None, multi_device=False
):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config", overrides=["agent=multi_objective/morld_ppo"]
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            pop_size=2,
            archive_capacity=4,
            total_timesteps=12,
            exchange_every=8,
            transfer=transfer,
            weight_adaptation=adaptation,
            target_workflow=dict(
                num_envs=2,
                num_eval_envs=2,
                eval_episodes=2,
                rollout_length=2,
                minibatch_size=4,
                reuse_rollout_epochs=1,
                agent_network=dict(
                    actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
                ),
            ),
        ),
    )

    def create_env(config, parallel, autoreset_mode, **kwargs):
        env = ActionSquashWrapper(ToyMultiObjectiveEnv(True, objectives))
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

    cls = type("TestMORLD", (MORLDPPOWorkflow,), {})
    with patch(
        "evorl.algorithms.multi_objective.moppo.create_env", side_effect=create_env
    ):
        return cls.build_from_config(config, jit=jit, multi_device=multi_device)


def test_morld_data_parallel_round_robin_and_policy_transfer():
    workflow = make_morld(multi_device=True, adaptation="PSA")
    try:
        with jax.set_mesh(workflow.mesh):
            state = workflow.setup(jax.random.key(31))
            metric, state = workflow.step(state)
            assert np.isfinite(metric.loss)
            assert int(state.metrics.sampled_timesteps) == 8
            assert int(state.metrics.training_updates) == 2
            assert int(state.current_policy) == 1
            metric, state = workflow.step(state)
            assert np.isfinite(metric.loss)
            assert int(state.metrics.sampled_timesteps) == 16
            assert int(state.current_policy) == 0
            for leaf in jax.tree.leaves(state.population.agent_state.model_state):
                assert leaf.sharding.is_fully_replicated
            env_obs = state.population.env_state.obs
            if workflow.num_devices > 1:
                assert not env_obs.sharding.is_fully_replicated
            assert env_obs.shape[:2] == (2, 2)
            evaluation, _ = workflow.evaluate(state)
            assert np.isfinite(evaluation.episode_returns).all()
    finally:
        workflow.close()


def assert_tree_close(expected, actual):
    for x, y in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        np.testing.assert_allclose(x, y, atol=1e-6)


def test_morld_training_transfer_only_copies_actor_and_resets_optimizer(jit):
    workflow = make_morld(jit=jit)
    try:
        assert workflow.workflow.evaluator.discount == workflow.workflow.config.discount
        state = workflow.setup(jax.random.key(1))
        old_target = jax.tree.map(lambda x: x[1], state.population)
        metrics, updated = workflow.step(state)
        assert np.isfinite(metrics.loss)
        assert int(updated.metrics.sampled_timesteps) == 8
        # Two envs finish once; the next rollout contains their reset-only step.
        assert int(updated.metrics.sampled_episodes) == 2
        assert int(updated.metrics.training_updates) == 2
        assert int(updated.current_policy) == 1
        source, target = [
            jax.tree.map(lambda x, i=i: x[i], updated.population) for i in range(2)
        ]
        agent = workflow.workflow.agent
        source_models, target_models = (
            agent.get_models(source.agent_state),
            agent.get_models(target.agent_state),
        )
        assert_tree_close(
            nnx.state(source_models.policy), nnx.state(target_models.policy)
        )
        assert_tree_close(
            nnx.state(agent.get_models(old_target.agent_state).value),
            nnx.state(target_models.value),
        )
        assert_tree_close(old_target.env_state, target.env_state)
        assert_tree_close(
            old_target.agent_state.extra_state, target.agent_state.extra_state
        )
        reset_optimizer = nnx.unpack(
            nnx.Optimizer(target_models, workflow.workflow.optimizer, wrt=nnx.Param)
        )
        assert_tree_close(reset_optimizer, target.opt_state)
        assert int(state.metrics.sampled_timesteps) == 0
        evaluation, _ = workflow.evaluate(updated)
        assert evaluation.episode_returns.shape == (2, 2)
    finally:
        workflow.close()


@pytest.mark.parametrize("objectives", [1, 3])
@pytest.mark.parametrize("multi_device", [False, True])
def test_morld_vector_shape_and_psa_nested_vmap(objectives, multi_device, mesh_factory):
    if multi_device:
        mesh_factory()
    workflow = make_morld(
        objectives=objectives, adaptation="PSA", multi_device=multi_device
    )
    try:
        with jax.set_mesh(workflow.mesh) if multi_device else nullcontext():
            keys = jax.random.split(jax.random.key(2), 4).reshape(2, 2)
            state = jax.jit(jax.vmap(jax.vmap(workflow.setup)))(keys)
            train, state = jax.jit(jax.vmap(jax.vmap(workflow.step)))(state)
            evaluation, _ = jax.jit(jax.vmap(jax.vmap(workflow.evaluate)))(state)
        assert train.loss.shape == (2, 2)
        assert evaluation.episode_returns.shape == (2, 2, 2, objectives)
        assert bool(jnp.isfinite(train.loss).all())
        np.testing.assert_allclose(state.preferences.sum(-1), 1)
    finally:
        workflow.close()


@pytest.mark.parametrize("multi_device", [False, True])
def test_morld_learn_tail_budget_checkpoint_and_restored_next_step(
    tmp_path, multi_device, mesh_factory
):
    if multi_device:
        mesh_factory()
    workflow = make_morld(multi_device=multi_device)
    workflow.recorder, workflow.checkpoint_manager = Mock(), Mock()
    try:
        with jax.set_mesh(workflow.mesh) if multi_device else nullcontext():
            state = workflow.learn(workflow.setup(jax.random.key(3)))
        assert int(state.metrics.sampled_timesteps) == 12
        assert int(state.metrics.training_updates) == 3
        assert int(state.metrics.iterations) == 2
        assert workflow.checkpoint_manager.save.call_args.kwargs["force"]
        assert workflow.recorder.write.called
        path = tmp_path / "morld"
        save(path, state)
        restored = load(path, state)
        with jax.set_mesh(workflow.mesh) if multi_device else nullcontext():
            _, expected = workflow.step(state)
            _, actual = workflow.step(restored)
        assert_tree_close(expected, actual)
        workflow.recorder.reset_mock()
        workflow.learn(restored)
        workflow.recorder.write.assert_not_called()
    finally:
        workflow.close()


@pytest.mark.parametrize(
    "environment, multi_device",
    [
        ("playground-cartpole", False),
        ("mo-mountaincar", False),
        ("playground-cartpole", True),
    ],
)
def test_morld_real_environment_training(environment, request, multi_device):
    if environment == "mo-mountaincar":
        request.getfixturevalue("mo_gymnasium_pools")
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config",
            overrides=[f"agent=exp/multi_objective/morld_ppo/{environment}"],
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            pop_size=2,
            archive_capacity=4,
            total_timesteps=4,
            exchange_every=4,
            target_workflow=dict(
                num_envs=2,
                num_eval_envs=2,
                eval_episodes=2,
                rollout_length=2,
                minibatch_size=4,
                reuse_rollout_epochs=1,
                agent_network=dict(
                    actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
                ),
            ),
        ),
    )
    cls = type("RealMORLD", (MORLDPPOWorkflow,), {})
    workflow = cls.build_from_config(config, multi_device=multi_device)
    workflow.recorder, workflow.checkpoint_manager = Mock(), Mock()
    try:
        with jax.set_mesh(workflow.mesh) if multi_device else nullcontext():
            try:
                state = workflow.setup(jax.random.key(9))
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
            with patch.object(
                workflow, "evaluate", wraps=workflow.evaluate
            ) as evaluate:
                state = workflow.learn(state)
                evaluate.assert_called_once()
            assert int(state.metrics.sampled_timesteps) == 4
            evaluation, _ = workflow.evaluate(state)
            assert evaluation.episode_returns.shape == (
                2,
                workflow.workflow.agent.num_objectives,
            )
            assert np.isfinite(evaluation.episode_returns).all()
            assert np.all(np.asarray(evaluation.episode_lengths) > 0)
            # Final logging must evaluate the transferred population, not the
            # pre-cooperation snapshots used by the archive and PSA.
            logged = [
                call.args[0]
                for call in workflow.recorder.write.call_args_list
                if "eval/episode_returns" in call.args[0]
            ]
            assert len(logged) == 1
            np.testing.assert_allclose(
                logged[0]["eval/episode_returns"], state.evaluation.episode_returns
            )
    finally:
        workflow.close()
