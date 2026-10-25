"""MARL workflows retain per-agent rewards, masks and termination semantics."""

from pathlib import Path
from unittest.mock import Mock, patch

from flax import nnx
from hydra.utils import get_class
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.envs import AutoresetMode
from evorl.envs.wrappers import (
    EpisodeWrapper,
    OneEpisodeWrapper,
    VmapAutoResetWrapper,
    VmapEnvPoolAutoResetWrapper,
    VmapWrapper,
)
from tests.helpers.multi_agent_envs import ToyMultiAgentEnv

pytestmark = pytest.mark.integration


def make_workflow(
    algorithm,
    jit,
    continuous=False,
    multi_device=False,
    normalize_obs=False,
    overrides=None,
):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        base_config = compose(
            config_name="config", overrides=[f"agent=multi_agents/{algorithm}"]
        )
    OmegaConf.set_struct(base_config, False)
    config = OmegaConf.merge(
        base_config,
        dict(
            env=dict(max_episode_steps=3),
            num_envs=4,
            num_eval_envs=4,
            eval_episodes=4,
            rollout_length=4,
            minibatch_size=8,
            reuse_rollout_epochs=1,
            normalize_obs=normalize_obs,
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
    )
    if algorithm in ("maddpg", "matd3"):
        config.random_timesteps = 8
        config.learning_start_timesteps = 8
        config.batch_size = 4
        config.replay_buffer_capacity = 64
        config.num_updates_per_iter = 2
    if overrides is not None:
        config = OmegaConf.merge(config, overrides)
    base = get_class(config.workflow_cls)
    cls = type(f"Test{algorithm}Workflow", (base,), {})

    def create_env(config, parallel, autoreset_mode, **kwargs):
        native = ToyMultiAgentEnv(continuous)
        if autoreset_mode == AutoresetMode.DISABLED:
            return VmapWrapper(
                OneEpisodeWrapper(native, config.max_episode_steps),
                parallel,
                vmap_step=True,
            )
        wrapper = (
            VmapEnvPoolAutoResetWrapper
            if algorithm in ("ippo", "mappo")
            else VmapAutoResetWrapper
        )
        assert autoreset_mode == (
            AutoresetMode.ENVPOOL
            if algorithm in ("ippo", "mappo")
            else AutoresetMode.NORMAL
        )
        return wrapper(
            EpisodeWrapper(
                native,
                config.max_episode_steps,
                record_ori_obs=autoreset_mode != AutoresetMode.ENVPOOL,
            ),
            parallel,
        )

    module = "maddpg" if algorithm in ("maddpg", "matd3") else "ippo"
    with patch(
        f"evorl.algorithms.multi_agents.{module}.create_env", side_effect=create_env
    ):
        return cls.build_from_config(config, multi_device=multi_device, jit=jit)


@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
@pytest.mark.parametrize("continuous", [False, True])
def test_ppo_multi_agent_workflow_updates_and_evaluates(algorithm, continuous, jit):
    workflow = make_workflow(algorithm, jit, continuous, normalize_obs=True)
    try:
        state = workflow.setup(jax.random.key(0))
        original = jax.tree.map(np.array, state.agent_state)
        metrics, updated = workflow.step(state)
        assert np.isfinite(metrics.loss)
        assert int(updated.metrics.sampled_timesteps) == 16
        assert int(updated.metrics.sampled_episodes) == 4
        assert set(metrics.raw_loss_dict) == {"first", "second"}
        models = nnx.merge(updated.agent_state.model_state, copy=True)
        initial = nnx.merge(state.agent_state.model_state, copy=True)
        for name in workflow.agent.agents:
            assert any(
                not np.array_equal(a, b)
                for a, b in zip(
                    jax.tree.leaves(nnx.state(models[name], nnx.Param)),
                    jax.tree.leaves(nnx.state(initial[name], nnx.Param)),
                    strict=True,
                )
            )
        evaluation, _ = workflow.evaluate(updated)
        np.testing.assert_allclose(evaluation.episode_returns.first, 2)
        np.testing.assert_allclose(evaluation.episode_returns.second, -6)
        np.testing.assert_allclose(evaluation.episode_lengths, 3)
        for before, after in zip(
            jax.tree.leaves(original), jax.tree.leaves(state.agent_state), strict=True
        ):
            np.testing.assert_array_equal(before, after)
    finally:
        workflow.close()


@pytest.mark.parametrize("updates_per_iter", [1, 3])
def test_matd3_delay_counts_critic_updates_across_iterations(updates_per_iter):
    workflow = make_workflow(
        "matd3",
        jit=True,
        continuous=True,
        overrides=dict(num_updates_per_iter=updates_per_iter, policy_delay=2),
    )
    try:
        state = workflow.setup(jax.random.key(17))
        for iteration in range(1, 3):
            previous = workflow.agent.get_models(state.agent_state, copy=True)
            metrics, state = workflow.step(state)
            models = workflow.agent.get_models(state.agent_state, copy=True)
            expected_updates = iteration * updates_per_iter // 2
            for name in workflow.agent.agents:
                assert (
                    int(nnx.merge(state.opt_state[name].actor, copy=True).step[...])
                    == expected_updates
                )
                assert (
                    int(nnx.merge(state.opt_state[name].critic, copy=True).step[...])
                    == iteration * updates_per_iter
                )
                if updates_per_iter == 1:
                    for target_name, source_name in (
                        ("target_actor", "actor"),
                        ("target_critic", "critic"),
                    ):
                        old_target = nnx.state(previous[name][target_name], nnx.Param)
                        new_target = nnx.state(models[name][target_name], nnx.Param)
                        source = nnx.state(models[name][source_name], nnx.Param)
                        for old, new, online in zip(
                            jax.tree.leaves(old_target),
                            jax.tree.leaves(new_target),
                            jax.tree.leaves(source),
                            strict=True,
                        ):
                            expected = (
                                (1 - workflow.config.tau) * old
                                + workflow.config.tau * online
                                if iteration == 2
                                else old
                            )
                            np.testing.assert_allclose(new, expected, atol=1e-7)
                    assert int(metrics.raw_loss_dict[name].actor_updates) == (
                        iteration // 2
                    )
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["ippo", "mappo", "maddpg", "matd3"])
def test_multi_agent_learn_records_agent_returns_and_stops_at_budget(algorithm):
    offpolicy = algorithm in ("maddpg", "matd3")
    budget = 40 if offpolicy else 32
    workflow = make_workflow(
        algorithm,
        jit=True,
        continuous=offpolicy,
        overrides=dict(total_timesteps=budget, fold_iters=1, eval_interval=100),
    )
    workflow.recorder = Mock()
    workflow.checkpoint_manager = Mock()
    try:
        state = workflow.init(jax.random.key(11))
        assert workflow.recorder.write.call_count == 0
        state = workflow.learn(state)
        assert int(state.metrics.iterations) == 2
        assert int(state.metrics.sampled_timesteps) == budget
        evaluation = workflow.recorder.write.call_args.args[0]
        assert set(evaluation["eval/episode_returns"]) == {"first", "second"}
        assert workflow.recorder.write.call_args.args[1] == 2
        assert workflow.checkpoint_manager.save.call_args.kwargs["force"] is True
        # Resuming a completed budget does not run another optimizer update.
        workflow.recorder.reset_mock()
        resumed = workflow.learn(state)
        assert int(resumed.metrics.iterations) == 2
        workflow.recorder.write.assert_not_called()
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["maddpg", "matd3"])
def test_offpolicy_multi_agent_workflow_preserves_joint_replay_and_targets(
    algorithm, jit
):
    workflow = make_workflow(
        algorithm,
        jit,
        continuous=True,
        normalize_obs=True,
        overrides=dict(random_timesteps=12, learning_start_timesteps=12),
    )
    try:
        state = workflow.setup(jax.random.key(0))
        assert int(state.metrics.sampled_timesteps) == 12
        assert int(state.metrics.sampled_episodes) == 0
        # Prefill did finish episodes, and their transitions remain in replay.
        assert int(state.replay_buffer_state.data.extras.episode_done[:12].sum()) == 4
        np.testing.assert_array_equal(state.env_state.env_state, 0)
        original = jax.tree.map(np.array, (state.agent_state, state.opt_state))
        metrics, updated = workflow.step(state)
        assert np.isfinite(metrics.loss)
        assert int(updated.metrics.sampled_timesteps) == 28
        assert int(updated.metrics.sampled_episodes) == 4
        assert int(updated.replay_buffer_state.buffer_size) == 28
        assert set(updated.replay_buffer_state.data.rewards) == {"first", "second"}
        for name in workflow.agent.agents:
            assert (
                int(nnx.merge(updated.opt_state[name].critic, copy=True).step[...]) == 2
            )
            assert int(
                nnx.merge(updated.opt_state[name].actor, copy=True).step[...]
            ) == (1 if algorithm == "matd3" else 2)
        evaluation, _ = workflow.evaluate(updated)
        np.testing.assert_allclose(evaluation.episode_returns.first, 2)
        np.testing.assert_allclose(evaluation.episode_returns.second, -6)
        for before, after in zip(
            jax.tree.leaves(original),
            jax.tree.leaves((state.agent_state, state.opt_state)),
            strict=True,
        ):
            np.testing.assert_array_equal(before, after)
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["maddpg", "matd3"])
def test_offpolicy_joint_replay_preserves_boolean_done_flags(algorithm, monkeypatch):
    original_reset, original_step = ToyMultiAgentEnv.reset, ToyMultiAgentEnv.step

    def boolean_flags(state):
        return state.replace(
            done=jax.tree.map(lambda x: x.astype(jnp.bool_), state.done),
            _internal=state._internal.replace(
                termination=jax.tree.map(
                    lambda x: x.astype(jnp.bool_), state._internal.termination
                ),
                truncation=jax.tree.map(
                    lambda x: x.astype(jnp.bool_), state._internal.truncation
                ),
            ),
        )

    monkeypatch.setattr(
        ToyMultiAgentEnv,
        "reset",
        lambda self, key: boolean_flags(original_reset(self, key)),
    )
    monkeypatch.setattr(
        ToyMultiAgentEnv,
        "step",
        lambda self, state, action: boolean_flags(original_step(self, state, action)),
    )
    workflow = make_workflow(algorithm, jit=True, continuous=True)
    try:
        state = workflow.setup(jax.random.key(29))
        metrics, state = workflow.step(state)
        assert np.isfinite(metrics.loss)
        for name in workflow.agent.agents:
            assert state.replay_buffer_state.data.dones[name].dtype == jnp.bool_
            assert (
                state.replay_buffer_state.data.extras.terminations[name].dtype
                == jnp.bool_
            )
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["ippo", "mappo", "maddpg", "matd3"])
def test_multi_agent_workflows_run_with_two_data_parallel_devices(algorithm, dp_mesh):
    workflow = make_workflow(
        algorithm,
        jit=True,
        continuous=algorithm in ("maddpg", "matd3"),
        multi_device=True,
    )
    try:
        with jax.set_mesh(workflow.mesh):
            state = workflow.setup(jax.random.key(2))
            metrics, updated = workflow.step(state)
            evaluation, _ = workflow.evaluate(updated)
        assert np.isfinite(metrics.loss)
        np.testing.assert_allclose(evaluation.episode_returns.first, 2)
        expected_samples = 24 if algorithm in ("maddpg", "matd3") else 16
        assert int(updated.metrics.sampled_timesteps) == expected_samples
        for variable in jax.tree.leaves(updated.agent_state):
            assert variable.sharding.is_fully_replicated
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["ippo", "mappo", "maddpg", "matd3"])
def test_multi_agent_workflows_train_on_real_jaxmarl_mpe(algorithm):
    pytest.importorskip("jaxmarl")
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config",
            overrides=[f"agent=exp/multi_agents/{algorithm}-mpe-spread"],
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
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
    )
    if algorithm in ("maddpg", "matd3"):
        config.random_timesteps = 4
        config.learning_start_timesteps = 4
        config.batch_size = 4
        config.replay_buffer_capacity = 32
        config.num_updates_per_iter = 2
    base = get_class(config.workflow_cls)
    cls = type(f"Real{algorithm}Workflow", (base,), {})
    workflow = cls.build_from_config(config, jit=True)
    try:
        state = workflow.setup(jax.random.key(4))
        metrics, updated = workflow.step(state)
        metrics2, updated = workflow.step(updated)
        evaluation, _ = workflow.evaluate(updated)
        assert np.isfinite(metrics.loss) and np.isfinite(metrics2.loss)
        assert set(evaluation.episode_returns) == set(workflow.env.agents)
        assert all(np.isfinite(x) for x in evaluation.episode_returns.values())
        assert int(updated.metrics.iterations) == 2
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
def test_multi_agent_ppo_with_real_jumanji_structured_observations(algorithm):
    pytest.importorskip("jumanji")
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(
            config_name="config",
            overrides=[
                f"agent=multi_agents/{algorithm}",
                "env=jumanji/routing/levelbasedforaging-v0",
            ],
        )
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(
        config,
        dict(
            env=dict(max_episode_steps=3),
            num_envs=2,
            num_eval_envs=2,
            eval_episodes=2,
            rollout_length=4,
            minibatch_size=4,
            reuse_rollout_epochs=1,
            normalize_obs=True,
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
    )
    base = get_class(config.workflow_cls)
    cls = type(f"Jumanji{algorithm}Workflow", (base,), {})
    workflow = cls.build_from_config(config, jit=True)
    try:
        state = workflow.setup(jax.random.key(23))
        metrics, state = workflow.step(state)
        evaluation, _ = workflow.evaluate(state)
        assert np.isfinite(metrics.loss)
        assert int(state.metrics.sampled_timesteps) == 8
        assert set(evaluation.episode_returns) == set(workflow.env.agents)
        assert all(np.isfinite(x) for x in evaluation.episode_returns.values())
        np.testing.assert_allclose(evaluation.episode_lengths, 3)
    finally:
        workflow.close()


@pytest.mark.parametrize("algorithm", ["ippo", "mappo", "maddpg", "matd3"])
def test_nested_populations_match_independent_multi_agent_workflows(algorithm):
    workflow = make_workflow(
        algorithm, jit=False, continuous=algorithm in ("maddpg", "matd3")
    )
    try:

        def train(key):
            state = workflow.setup(key)
            return workflow.step(state)

        keys = jax.random.split(jax.random.key(8), (2, 2))
        # Compare batching semantics at full float32 matmul precision; GPU
        # batched and unbatched GEMMs may otherwise choose different precision.
        with jax.default_matmul_precision("highest"):
            metrics, population = jax.jit(jax.vmap(jax.vmap(train)))(keys)
            expected_metrics, expected_state = jax.jit(train)(keys[0, 0])
        assert np.all(np.isfinite(metrics.loss))
        for actual, expected in zip(
            jax.tree.leaves(population.agent_state),
            jax.tree.leaves(expected_state.agent_state),
            strict=True,
        ):
            np.testing.assert_allclose(actual[0, 0], expected, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(
            metrics.loss[0, 0], expected_metrics.loss, atol=2e-6, rtol=2e-5
        )
    finally:
        workflow.close()
