"""Exercise real PPO setup, training and evaluation with Jumanji observations."""

from pathlib import Path

import jax
import numpy as np
import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf

from evorl.algorithms.random_agent import RandomAgentWorkflow

pytest.importorskip("jumanji")
pytestmark = pytest.mark.integration


def test_ppo_jumanji_under_nested_vmap():
    root = Path(__file__).resolve().parents[2]
    env_config = OmegaConf.load(
        root / "configs/env/jumanji/logic/graphcoloring-v1.yaml"
    )
    env_config.max_episode_steps = 3
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / "configs/agent/ppo.yaml"),
        dict(
            env=env_config,
            num_envs=2,
            num_eval_envs=2,
            eval_episodes=2,
            rollout_length=4,
            minibatch_size=4,
            reuse_rollout_epochs=1,
            agent_network=dict(
                actor_hidden_layer_sizes=[8],
                critic_hidden_layer_sizes=[8],
                policy_obs_key="colors",
                value_obs_key="colors",
            ),
        ),
    )
    base = get_class(config.workflow_cls)
    cls = type("JumanjiPPO", (base,), {})
    workflow = cls.build_from_config(config, jit=False)
    try:
        keys = jax.random.split(jax.random.key(0), (2, 2))
        state = jax.jit(jax.vmap(jax.vmap(workflow.setup)))(keys)
        assert state.env_state.obs.colors.shape == (2, 2, 2, 20)
        before = state.agent_state.model_state
        metrics, state = jax.jit(jax.vmap(jax.vmap(workflow.step)))(state)
        jax.block_until_ready(state)
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
        metrics, state = jax.jit(jax.vmap(jax.vmap(workflow.evaluate)))(state)
        assert metrics.episode_returns.shape == (2, 2)
        assert np.isfinite(metrics.episode_returns).all()
        assert np.all(metrics.episode_lengths > 0)
        assert np.all(metrics.episode_lengths <= 3)
    finally:
        workflow.close()


def test_random_agent_with_integer_vector_actions():

    root = Path(__file__).resolve().parents[2]
    env_config = OmegaConf.load(root / "configs/env/jumanji/logic/sudoku-v0.yaml")
    env_config.max_episode_steps = 2
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        dict(
            env=env_config,
            num_eval_envs=2,
            eval_episodes=2,
        ),
    )
    cls = type("JumanjiRandom", (RandomAgentWorkflow,), {})
    workflow = cls.build_from_config(config, jit=False)
    try:
        state = jax.jit(workflow.setup)(jax.random.key(0))
        metrics, state = jax.jit(workflow.evaluate)(state)
        assert np.isfinite(metrics.episode_returns).all()
        assert np.all(metrics.episode_lengths > 0)
        assert np.all(metrics.episode_lengths <= 2)
    finally:
        workflow.close()
