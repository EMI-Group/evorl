"""Independent TD3 populations with a shared replay buffer."""

from pathlib import Path
from unittest.mock import patch

from flax import nnx
import jax
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.algorithms.contrib.pop_td3 import PopTD3Workflow
from tests.helpers.envs import ToyEnv

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "random_timesteps,learning_start_timesteps", [(-1, 4), (0, -1), (16, 4)]
)
def test_invalid_warmup_is_rejected_before_environment_creation(
    random_timesteps, learning_start_timesteps
):
    config = OmegaConf.create(
        dict(
            random_timesteps=random_timesteps,
            learning_start_timesteps=learning_start_timesteps,
        )
    )
    cls = type("InvalidWarmupPopTD3", (PopTD3Workflow,), {})
    with patch("evorl.algorithms.td3.create_env") as create_env:
        with pytest.raises(
            ValueError, match="0 <= random_timesteps <= learning_start_timesteps"
        ):
            cls.build_from_config(config, jit=False)
        create_env.assert_not_called()


@pytest.mark.parametrize("norm", ["none", "spectral_norm"])
@pytest.mark.parametrize("normalize_obs", [False, True])
def test_population_step_preserves_members_and_updates_native_optimizers(
    norm, normalize_obs, jit
):
    root = Path(__file__).resolve().parents[2]
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / "configs/agent/td3.yaml"),
        OmegaConf.load(root / "configs/agent/contrib/pop-td3.yaml"),
        dict(
            env=dict(max_episode_steps=2),
            pop_size=3,
            num_envs=2,
            num_eval_envs=2,
            eval_episodes=2,
            random_timesteps=4,
            learning_start_timesteps=10,
            batch_size=2,
            replay_buffer_capacity=32,
            rollout_length=2,
            actor_update_interval=2,
            num_updates_per_iter=1,
            normalize_obs=normalize_obs,
            agent_network=dict(
                actor_hidden_layer_sizes=[4],
                critic_hidden_layer_sizes=[4],
                norm_layer_type=norm,
            ),
        ),
    )
    cls = type("PopulationTD3TestWorkflow", (PopTD3Workflow,), {})
    with (
        patch("jax.local_devices", return_value=jax.local_devices()[:1]),
        patch(
            "evorl.algorithms.td3.create_env",
            side_effect=lambda _, parallel, **kw: ToyEnv(parallel),
        ),
    ):
        workflow = cls.build_from_config(config, jit=jit)
        state = workflow.setup(jax.random.key(8))
    try:
        original = [
            np.array(x, copy=True)
            for x in jax.tree.leaves((state.agent_state, state.opt_state))
        ]
        metrics, updated = workflow.step(state)
        jax.block_until_ready(updated)
        for before, after in zip(
            original, jax.tree.leaves((state.agent_state, state.opt_state)), strict=True
        ):
            np.testing.assert_array_equal(before, after)
        assert int(updated.metrics.iterations) == 1
        assert np.isfinite(metrics.critic_loss).all()
        assert np.isfinite(metrics.actor_loss).all()
        assert int(updated.replay_buffer_state.buffer_size) == 22
        for name, expected_count in (("actor", 1), ("critic", 2)):
            optimizer = nnx.merge(updated.opt_state[name], copy=True)
            np.testing.assert_array_equal(optimizer.step[...], [expected_count] * 3)
        initial_models = nnx.merge(state.agent_state.model_state, copy=True)
        updated_models = nnx.merge(updated.agent_state.model_state, copy=True)
        for name in ("actor", "critic"):
            assert any(
                not np.array_equal(before, after)
                for before, after in zip(
                    jax.tree.leaves(nnx.state(initial_models[name], nnx.Param)),
                    jax.tree.leaves(nnx.state(updated_models[name], nnx.Param)),
                    strict=True,
                )
            )
        evaluation, _ = workflow.evaluate(updated)
        assert evaluation.episode_return.shape == (3,)
        assert np.isfinite(evaluation.episode_return).all()
        if normalize_obs:
            np.testing.assert_array_equal(
                updated.agent_state.obs_preprocessor_state.count,
                state.agent_state.obs_preprocessor_state.count + 12,
            )
    finally:
        workflow.close()
