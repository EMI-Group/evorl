"""Train and evaluate real workflows with nested host-environment batches."""

from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np
import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf

pytestmark = pytest.mark.integration


def test_ppo_nested_host_environments(host_envs):
    backend, environments = host_envs
    root = Path(__file__).resolve().parents[2]
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / "configs/agent/ppo.yaml"),
        dict(
            env=dict(env_type=backend, env_name="CartPole-v1", max_episode_steps=3),
            num_envs=1,
            num_eval_envs=1,
            eval_episodes=1,
            rollout_length=8,
            minibatch_size=4,
            reuse_rollout_epochs=1,
            normalize_obs=True,
            agent_network=dict(
                actor_hidden_layer_sizes=[8], critic_hidden_layer_sizes=[8]
            ),
        ),
    )
    base = get_class(config.workflow_cls)
    cls = type(f"NestedHost{base.__name__}", (base,), {})
    workflow = cls.build_from_config(config, jit=False)
    try:
        keys = jax.random.split(jax.random.key(0), 4).reshape(2, 2)
        setup = jax.jit(jax.vmap(jax.vmap(workflow.setup)))
        step = jax.jit(jax.vmap(jax.vmap(workflow.step)))
        evaluate = jax.jit(jax.vmap(jax.vmap(workflow.evaluate)))
        state = setup(keys)
        assert state.env_state.obs.shape == (2, 2, 1, 4)
        before = state.agent_state.model_state
        metrics, state = step(state)
        jax.block_until_ready(state)
        assert state.metrics.iterations.shape == (2, 2)
        np.testing.assert_array_equal(state.metrics.iterations, np.ones((2, 2)))
        assert any(
            np.any(np.asarray(a) != np.asarray(b))
            for a, b in zip(
                jax.tree.leaves(before),
                jax.tree.leaves(state.agent_state.model_state),
                strict=True,
            )
        )
        for leaf in jax.tree.leaves(metrics):
            assert np.isfinite(leaf).all()
        metrics, state = evaluate(state)
        jax.block_until_ready(state)
        assert metrics.episode_returns.shape == (2, 2)
        assert np.isfinite(metrics.episode_returns).all()
        assert np.all(metrics.episode_lengths > 0)
        assert np.all(metrics.episode_lengths <= 3)
        assert environments
    finally:
        workflow.close()


def test_pbt_openes_host_environments(host_envs):
    if jax.local_device_count() != 1:
        pytest.skip(
            "Host pools use one device; run with --cpu-devices=1 or --device=gpu"
        )
    backend, environments = host_envs
    root = Path(__file__).resolve().parents[2]
    common = OmegaConf.load(root / "configs/config.yaml")
    target = OmegaConf.merge(
        common,
        OmegaConf.load(root / "configs/agent/meta/paramopenes.yaml"),
        dict(
            pop_size=4,
            num_envs=1,
            episodes_for_fitness=1,
            num_eval_envs=1,
            eval_episodes=1,
            normalize_obs=False,
            agent_network=dict(actor_hidden_layer_sizes=[8]),
        ),
    )
    config = OmegaConf.merge(
        common,
        OmegaConf.load(root / "configs/agent/meta/pbt-paramopenes.yaml"),
        dict(
            target_workflow=target,
            env=dict(env_type=backend, env_name="Pendulum-v1", max_episode_steps=3),
            pop_size=4,
            num_eval_envs=1,
            eval_episodes=1,
            workflow_steps_per_iter=1,
            warmup_steps=100,
            bottom_ratio=0.25,
            top_ratio=0.25,
        ),
    )
    target_base = get_class(target.workflow_cls)
    target_cls = type(f"Host{target_base.__name__}", (target_base,), {})
    base = get_class(config.workflow_cls)
    cls = type(f"Host{base.__name__}", (base,), {})
    with patch("hydra.utils.get_class", return_value=target_cls):
        workflow = cls.build_from_config(config)
    try:
        with jax.set_mesh(workflow.mesh):
            state = workflow.setup(jax.random.key(0))
            before = state.pop_workflow_state.ec_opt_state.mean
            metrics, state = workflow.step(state)
            jax.block_until_ready(state)
            assert metrics.pop_episode_returns.shape == (4,)
            assert np.isfinite(metrics.pop_episode_returns).all()
            assert any(
                np.any(np.asarray(a) != np.asarray(b))
                for a, b in zip(
                    jax.tree.leaves(before),
                    jax.tree.leaves(state.pop_workflow_state.ec_opt_state.mean),
                    strict=True,
                )
            )
            # Four PBT members each evaluate four OpenES candidates in a
            # single native pool: these are two real population vmap axes.
            assert workflow.workflow.env.unwrapped.env.num_envs == 16
            metrics, state = workflow.evaluate(state)
            jax.block_until_ready(state)
            assert metrics.pop_episode_returns.shape == (4,)
            assert np.isfinite(metrics.pop_episode_returns).all()
            assert np.all(metrics.pop_episode_lengths == 3)
            assert environments
    finally:
        workflow.close()


@pytest.mark.parametrize("host_envs", ["gymnasium"], indirect=True)
def test_pbt_sac_restores_population_host_batches_after_shared_prefill(host_envs):
    if jax.local_device_count() != 1:
        pytest.skip(
            "Host pools use one device; run with --cpu-devices=1 or --device=gpu"
        )
    backend, _ = host_envs
    root = Path(__file__).resolve().parents[2]
    common = OmegaConf.load(root / "configs/config.yaml")
    target = OmegaConf.merge(
        common,
        OmegaConf.load(root / "configs/agent/meta/paramsac.yaml"),
        dict(
            num_envs=2,
            num_eval_envs=1,
            eval_episodes=1,
            rollout_length=1,
            batch_size=2,
            replay_buffer_capacity=16,
            num_updates_per_iter=1,
            normalize_obs=False,
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
    )
    config = OmegaConf.merge(
        common,
        OmegaConf.load(root / "configs/agent/meta/pbt-paramsac.yaml"),
        dict(
            target_workflow=target,
            env=dict(env_type=backend, env_name="Pendulum-v1", max_episode_steps=100),
            pop_size=2,
            random_timesteps=6,
            shared_replay_buffer=True,
            num_eval_envs=1,
            eval_episodes=1,
            workflow_steps_per_iter=1,
            warmup_steps=100,
            bottom_ratio=0.5,
            top_ratio=0.5,
        ),
    )
    target_cls = type("HostParamSAC", (get_class(target.workflow_cls),), {})
    cls = type("HostPBTSAC", (get_class(config.workflow_cls),), {})
    with patch("hydra.utils.get_class", return_value=target_cls):
        workflow = cls.build_from_config(config, multi_device=False)
    try:
        with jax.set_mesh(workflow.mesh):
            state = workflow.setup(jax.random.key(20))
            jax.block_until_ready(state)
            assert workflow.workflow.env.env.num_envs == 4
            np.testing.assert_array_equal(
                state.pop_workflow_state.env_state.info.steps, 0
            )
            # Exercise the actual pool after restoring its population batch.
            native = workflow.workflow.env.env
            np.testing.assert_array_equal(
                native.call("get_wrapper_attr", "_elapsed_steps"), [0] * 4
            )
            observations, *_ = native.step(np.zeros((4, 1), np.float32))
            assert observations.shape == (4, 3)
            np.testing.assert_array_equal(
                native.call("get_wrapper_attr", "_elapsed_steps"), [1] * 4
            )
            np.testing.assert_array_equal(state.replay_buffer_state.buffer_size, 6)
    finally:
        workflow.close()
