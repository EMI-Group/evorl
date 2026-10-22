"""TD3's independent critics and delayed actor/target update regression tests."""

from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np
import pytest
from omegaconf import OmegaConf
from flax import nnx

from evorl.algorithms.contrib.td3_v3 import TD3V3Workflow
from tests.helpers.envs import ToyEnv


@pytest.mark.parametrize(
    "norm", ["none", "layer_norm", "static_layer_norm", "spectral_norm"]
)
def test_independent_critics_and_delayed_target_updates(norm, jit):
    root = Path(__file__).resolve().parents[2]
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / "configs/agent/contrib/td3-v3.yaml"),
        {
            "env": {"max_episode_steps": 2},
            "num_envs": 2,
            "num_eval_envs": 2,
            "random_timesteps": 4,
            "learning_start_timesteps": 4,
            "batch_size": 4,
            "replay_buffer_capacity": 32,
            "rollout_length": 1,
            "normalize_obs": True,
            "tau": 0.25,
            "policy_noise": 0.0,
            "actor_update_interval": 2,
            "optimizer": {"lr": 0.01},
            "agent_network": {
                "actor_hidden_layer_sizes": [4],
                "critic_hidden_layer_sizes": [4],
                "norm_layer_type": norm,
            },
        },
    )
    workflow_cls = type("TD3V3TestWorkflow", (TD3V3Workflow,), {})
    with patch(
        "evorl.algorithms.contrib.td3_v3.create_env",
        side_effect=lambda _, parallel, **kw: ToyEnv(parallel),
    ):
        workflow = workflow_cls.build_from_config(config, jit=jit)
    try:
        state = workflow.init(jax.random.key(42))
        initial = jax.tree.map(
            lambda x: np.array(x, copy=True), state.agent_state.model_state
        )
        metrics1, first = workflow.step(state)
        jax.block_until_ready(first)
        assert int(first.metrics.iterations) == 1
        assert np.isfinite(metrics1.critic1_loss)
        assert np.isfinite(metrics1.critic2_loss)
        for before, after in zip(
            jax.tree.leaves(initial.actor),
            jax.tree.leaves(first.agent_state.model_state.actor),
            strict=True,
        ):
            np.testing.assert_array_equal(before, after)
        for name in (
            "target_actor",
            "target_critic1",
            "target_critic2",
        ):
            for before, after in zip(
                jax.tree.leaves(nnx.filter_state(getattr(initial, name), nnx.Param)),
                jax.tree.leaves(
                    nnx.filter_state(
                        getattr(first.agent_state.model_state, name), nnx.Param
                    )
                ),
                strict=True,
            ):
                np.testing.assert_array_equal(before, after)
        for name in ("critic1", "critic2"):
            assert any(
                not np.array_equal(before, after)
                for before, after in zip(
                    jax.tree.leaves(
                        nnx.filter_state(getattr(initial, name), nnx.Param)
                    ),
                    jax.tree.leaves(
                        nnx.filter_state(
                            getattr(first.agent_state.model_state, name), nnx.Param
                        )
                    ),
                    strict=True,
                )
            )
        metrics2, second = workflow.step(first)
        jax.block_until_ready(second)
        assert np.isfinite(metrics2.actor_loss)
        assert int(second.metrics.iterations) == 2
        assert any(
            not np.array_equal(before, after)
            for before, after in zip(
                jax.tree.leaves(first.agent_state.model_state.actor),
                jax.tree.leaves(second.agent_state.model_state.actor),
                strict=True,
            )
        )
        for online, target in (
            ("actor", "target_actor"),
            ("critic1", "target_critic1"),
            ("critic2", "target_critic2"),
        ):
            expected = jax.tree.map(
                lambda old, new: 0.75 * old + 0.25 * new,
                nnx.filter_state(
                    getattr(first.agent_state.model_state, target), nnx.Param
                ),
                nnx.filter_state(
                    getattr(second.agent_state.model_state, online), nnx.Param
                ),
            )
            for actual, desired in zip(
                jax.tree.leaves(
                    nnx.filter_state(
                        getattr(second.agent_state.model_state, target), nnx.Param
                    )
                ),
                jax.tree.leaves(expected),
                strict=True,
            ):
                np.testing.assert_allclose(actual, desired, rtol=2e-5, atol=1e-6)
        for before, after in zip(
            jax.tree.leaves(initial),
            jax.tree.leaves(state.agent_state.model_state),
            strict=True,
        ):
            np.testing.assert_array_equal(before, after)
    finally:
        workflow.close()
