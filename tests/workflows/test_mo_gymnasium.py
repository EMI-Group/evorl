"""Real MO-Gymnasium training and evaluation with independent nested populations."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class
import jax
import numpy as np
from omegaconf import OmegaConf
import pytest

pytest.importorskip("mo_gymnasium")

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "example", ["envelope_q-mo-deep-sea-treasure", "capql-mo-mountaincar"]
)
@pytest.mark.parametrize("nested", [False, True], ids=["single", "nested"])
def test_real_mo_workflow_updates_and_evaluates_vectors(
    mo_gymnasium_pools, example, nested
):
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
            num_eval_envs=1,
            eval_episodes=1,
            eval_preferences=[[1.0, 0.0], [0.0, 1.0]],
            rollout_length=2,
            random_timesteps=4,
            learning_start_timesteps=4,
            batch_size=2,
            replay_buffer_capacity=16,
            num_updates_per_iter=2,
            normalize_obs=True,
            agent_network=dict(
                actor_hidden_layer_sizes=[4],
                critic_hidden_layer_sizes=[4],
                q_hidden_layer_sizes=[4],
            ),
        ),
    )
    base = get_class(config.workflow_cls)
    cls = type(f"Host{base.__name__}", (base,), {})
    workflow = cls.build_from_config(config, jit=not nested)
    try:

        def run(key):
            state = workflow.setup(key)
            before = state.agent_state.model_state
            train_metrics, state = workflow.step(state)
            eval_metrics, state = workflow.evaluate(state)
            return before, state, train_metrics, eval_metrics

        if nested:
            fn = jax.jit(jax.vmap(jax.vmap(run)))
            keys = jax.random.split(jax.random.key(12), (2, 2))
            batch_shape = (2, 2)
        else:
            fn, keys, batch_shape = run, jax.random.key(12), ()
        before, state, training, evaluation = fn(keys)
        jax.block_until_ready(state)
        assert state.env_state.reward.shape == (*batch_shape, 2, 2)
        assert state.replay_buffer_state.data.rewards.shape == (*batch_shape, 16, 2)
        np.testing.assert_array_equal(
            state.metrics.sampled_timesteps, np.full(batch_shape, 8)
        )
        np.testing.assert_array_equal(
            state.metrics.training_updates, np.full(batch_shape, 2)
        )
        assert np.isfinite(training.loss).all()
        assert evaluation.episode_returns.shape == (*batch_shape, 2, 2)
        assert evaluation.episode_lengths.shape == (*batch_shape, 2)
        assert np.isfinite(evaluation.episode_returns).all()
        assert np.all(np.asarray(evaluation.episode_lengths) > 0)
        assert np.all(np.asarray(evaluation.episode_lengths) <= 3)
        np.testing.assert_allclose(
            evaluation.scalarized_returns,
            (np.asarray(evaluation.preferences) * evaluation.episode_returns).sum(-1),
            atol=1e-6,
        )
        assert any(
            not np.array_equal(old, new)
            for old, new in zip(
                jax.tree.leaves(before),
                jax.tree.leaves(state.agent_state.model_state),
                strict=True,
            )
        )
    finally:
        workflow.close()
