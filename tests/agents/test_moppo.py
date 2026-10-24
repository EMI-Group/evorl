"""MOPPO vector GAE, scalarization and continuous action likelihoods."""

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest
from flax import nnx

from evorl.algorithms.multi_objective.moppo import MOPPOAgent
from evorl.distributed import DP_AXIS_NAME, pmean_nnx_loss_and_grad
from evorl.envs.wrappers import ActionSquashWrapper
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.rl_toolkits import compute_gae
from tests.helpers.multi_objective_envs import ToyMultiObjectiveEnv


@pytest.mark.parametrize("valid_counts", [(2, 2), (1, 2), (0, 2), (0, 0)])
@pytest.mark.parametrize("normalize", [True, False])
def test_data_parallel_advantage_normalization_matches_combined_minibatch(
    valid_counts, normalize, dp_mesh
):
    env = ToyMultiObjectiveEnv()
    agent = MOPPOAgent(
        num_objectives=2,
        continuous_action=False,
        agent_network_cfg=OmegaConf.create(
            dict(actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4])
        ),
        normalize_gae=normalize,
    )
    state = agent.init(env.obs_space, env.action_space, jax.random.key(53))
    actions = jnp.array([0, 0, 1, 1])
    valid = np.concatenate([np.arange(2) < count for count in valid_counts])
    batch = SampleBatch(obs=jnp.ones((4, 3)), actions=actions)
    models = agent.get_models(state, copy=True)
    old_logp = jax.nn.log_softmax(models.policy(batch.obs))[jnp.arange(4), actions]
    batch = batch.replace(
        extras=PyTreeDict(
            env_extras=PyTreeDict(autoreset=jnp.asarray(~valid)),
            advantages=jnp.where(
                valid[:, None],
                jnp.array([[1.0, 1.0], [1.0, 1.0], [3.0, 3.0], [4.0, 4.0]]),
                1e6,
            ),
            policy_extras=PyTreeDict(logp=old_logp + jnp.where(valid, 0.0, 0.5)),
            values=agent.compute_values(state, batch),
            v_targets=jnp.where(valid[:, None], jnp.ones((4, 2)), 1e6),
        )
    )

    def loss(models, batch, axis_name=None, agent_state=state):
        losses = agent.loss(
            agent_state.replace(model_state=nnx.unpack(models)),
            batch,
            jax.random.key(0),
            axis_name=axis_name,
        )
        return (
            losses.actor_loss
            + 0.5 * losses.critic_loss
            - 0.01 * losses.actor_entropy
            + 0.01 * losses.approx_kl
        )

    reference = jax.tree.map(lambda x: x[valid], batch) if valid.any() else batch
    expected_loss, expected_grads = nnx.value_and_grad(loss)(models, reference)

    @jax.shard_map(in_specs=(jax.P(), jax.P(DP_AXIS_NAME)), out_specs=jax.P())
    def distributed(agent_state, batch):
        models = agent.get_models(agent_state, copy=True)
        value, grads = pmean_nnx_loss_and_grad(
            lambda models, batch: loss(models, batch, DP_AXIS_NAME, agent_state),
            DP_AXIS_NAME,
        )(models, batch)
        return value, grads

    actual_loss, actual_grads = jax.jit(distributed)(
        jax.device_put(state, jax.P()),
        jax.device_put(batch, jax.P(DP_AXIS_NAME)),
    )
    np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-6)
    if valid.any():
        assert (
            float(sum(jnp.square(x).sum() for x in jax.tree.leaves(expected_grads))) > 0
        )
    else:
        np.testing.assert_array_equal(actual_loss, 0)
    for actual, expected in zip(
        jax.tree.leaves(actual_grads), jax.tree.leaves(expected_grads), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        if not valid.any():
            np.testing.assert_array_equal(actual, 0)


@pytest.mark.parametrize("objectives", [2, 3])
def test_vector_gae_bootstraps_truncation_but_does_not_cross_reset(objectives):
    scale = jnp.arange(1, objectives + 1, dtype=jnp.float32)
    rewards = jnp.array([2.0, 0.0, 3.0, 0.0])[:, None, None] * scale
    # First episode truncates at t=0, t=1 resets, second terminates at t=2,
    # t=3 resets. Reset observations must not bootstrap either episode boundary.
    values = jnp.array([1.0, 5.0, 2.0, 7.0, 1000.0])[:, None, None] * scale
    done = jnp.array([1.0, 0.0, 1.0, 0.0])[:, None, None]
    terminated = jnp.array([0.0, 0.0, 1.0, 0.0])[:, None, None]
    targets, advantages = jax.jit(compute_gae)(
        rewards, values, done, terminated, discount=0.5, gae_lambda=1.0
    )
    np.testing.assert_allclose(
        advantages[jnp.array([0, 2]), 0], jnp.array([3.5, 1.0])[:, None] * scale
    )
    np.testing.assert_allclose(
        targets[jnp.array([0, 2]), 0], jnp.array([4.5, 3.0])[:, None] * scale
    )


@pytest.mark.parametrize("continuous", [False, True])
def test_moppo_behavior_ratio_and_preference_scalarization(continuous):
    env = ToyMultiObjectiveEnv(continuous)
    if continuous:
        env = ActionSquashWrapper(env)
    agent = MOPPOAgent(
        num_objectives=2,
        continuous_action=continuous,
        agent_network_cfg=OmegaConf.create(
            dict(actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4])
        ),
        normalize_gae=False,
        clip_value_loss=False,
    )
    state = agent.init(env.obs_space, env.action_space, jax.random.key(0))
    state = state.replace(
        extra_state=state.extra_state.replace(preferences=jnp.array([1.0, 0.0]))
    )
    batch = SampleBatch(obs=jnp.ones((4, 3)))
    actions, policy_extras = agent.compute_actions(state, batch, jax.random.key(1))
    if continuous:
        assert "action_scale" not in state.extra_state
        assert "action_bias" not in state.extra_state
        assert bool(jnp.all(actions >= env.action_space.low))
        assert bool(jnp.all(actions <= env.action_space.high))
    values = agent.compute_values(state, batch)
    batch = batch.replace(
        actions=actions,
        extras=PyTreeDict(
            env_extras=PyTreeDict(autoreset=jnp.zeros(4, dtype=jnp.bool_)),
            policy_extras=policy_extras,
            advantages=jnp.array([[1.0, 100.0]] * 4),
            values=values,
            v_targets=values,
        ),
    )
    loss = jax.jit(agent.loss)(state, batch, jax.random.key(2))
    np.testing.assert_allclose(loss.actor_loss, -1.0, atol=1e-5)
    np.testing.assert_allclose(loss.critic_loss, 0.0, atol=1e-7)
    np.testing.assert_allclose(loss.approx_kl, 0.0, atol=1e-6)
    switched = state.replace(
        extra_state=state.extra_state.replace(preferences=jnp.array([0.0, 1.0]))
    )
    np.testing.assert_allclose(
        agent.loss(switched, batch, jax.random.key(2)).actor_loss, -100.0, atol=1e-3
    )
