"""V-trace targets against independent recurrences and on-policy GAE."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from evorl.algorithms.impala import IMPALAAgent, compute_vtrace
from evorl.distributed import DP_AXIS_NAME, pmean_nnx_loss_and_grad
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.rl_toolkits import compute_gae
from tests.helpers.agents import build_agent
from tests.helpers.envs import ToyEnv


@pytest.mark.parametrize("valid_counts", [(4, 4), (2, 4), (0, 4), (0, 0)])
def test_masked_losses_and_gradients_match_global_sequences(valid_counts, dp_mesh):
    """Keep V-trace sequences intact while varying valid counts across devices."""
    env = ToyEnv(4, discrete=True)
    agent = build_agent(
        IMPALAAgent,
        env.obs_space,
        env.action_space,
        actor_hidden_layer_sizes=(4,),
        critic_hidden_layer_sizes=(4,),
        normalize_obs=False,
    )
    state = agent.init(env.obs_space, env.action_space, jax.random.key(61))
    obs = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) / 10
    models = agent.get_models(state, copy=True)
    logp = jax.nn.log_softmax(models.policy(obs))[..., 0]
    valid = np.concatenate(
        [(np.arange(4) < count).reshape(2, 2) for count in valid_counts], axis=1
    )
    batch = SampleBatch(
        obs=obs,
        next_obs=obs + 0.1,
        actions=jnp.zeros((2, 4), jnp.int32),
        rewards=jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        dones=jnp.zeros((2, 4)),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                autoreset=jnp.asarray(~valid),
                termination=jnp.zeros((2, 4)),
            ),
            policy_extras=PyTreeDict(logp=logp + 0.2),
        ),
    )

    def loss(models, state, batch, axis_name=None):
        losses = agent.loss(
            state.replace(model_state=nnx.unpack(models)),
            batch,
            jax.random.key(0),
            axis_name=axis_name,
        )
        return (
            losses.actor_loss + 0.5 * losses.critic_loss - 0.01 * losses.actor_entropy,
            losses,
        )

    expected, expected_grads = nnx.value_and_grad(loss, has_aux=True)(
        models, state, batch
    )

    @jax.jit
    @jax.shard_map(in_specs=(jax.P(), jax.P(None, DP_AXIS_NAME)), out_specs=jax.P())
    def distributed(state, batch):
        local_models = agent.get_models(state, copy=True)
        (value, losses), grads = pmean_nnx_loss_and_grad(
            lambda models: loss(models, state, batch, DP_AXIS_NAME),
            DP_AXIS_NAME,
            has_aux=True,
        )(local_models)
        return (value, jax.lax.pmean(losses, DP_AXIS_NAME)), grads

    actual, actual_grads = distributed(
        state, jax.device_put(batch, jax.P(None, DP_AXIS_NAME))
    )
    for value, reference in zip(
        jax.tree.leaves((actual, actual_grads)),
        jax.tree.leaves((expected, expected_grads)),
        strict=True,
    ):
        np.testing.assert_allclose(value, reference, rtol=1e-5, atol=1e-6)
        if not valid.any():
            np.testing.assert_array_equal(value, 0)


@pytest.mark.parametrize("empty_first_device", [False, True])
def test_kl_uses_valid_samples_across_time_and_devices(empty_first_device, dp_mesh):
    env = ToyEnv(4, discrete=True)
    agent = build_agent(
        IMPALAAgent,
        env.obs_space,
        env.action_space,
        actor_hidden_layer_sizes=(4,),
        critic_hidden_layer_sizes=(4,),
        normalize_obs=False,
    )
    state = agent.init(env.obs_space, env.action_space, jax.random.key(67))
    obs = jnp.ones((2, 4, 3))
    actions = jnp.zeros((2, 4), jnp.int32)
    logits = agent.get_models(state, copy=True).policy(obs)
    logp = jax.nn.log_softmax(logits)[..., 0]
    logratio = np.array([[0.1, 3.0, 0.2, -0.2], [0.3, -2.0, 0.4, 0.5]])
    valid = np.array([[True, False, True, True], [True, False, False, True]])
    if empty_first_device:
        valid[:, :2] = False
    batch = SampleBatch(
        obs=obs,
        next_obs=obs + 0.1,
        actions=actions,
        rewards=jnp.ones((2, 4)),
        dones=jnp.zeros((2, 4)),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                autoreset=jnp.asarray(~valid), termination=jnp.zeros((2, 4))
            ),
            policy_extras=PyTreeDict(logp=logp - jnp.asarray(logratio, jnp.float32)),
        ),
    )
    expected = (np.expm1(logratio[valid]) - logratio[valid]).mean()
    local = agent.loss(state, batch, jax.random.key(0)).approx_kl
    np.testing.assert_allclose(local, expected, rtol=1e-5, atol=1e-7)

    @jax.jit
    @jax.shard_map(in_specs=(jax.P(), jax.P(None, DP_AXIS_NAME)), out_specs=jax.P())
    def distributed(state, batch):
        value = agent.loss(
            state, batch, jax.random.key(0), axis_name=DP_AXIS_NAME
        ).approx_kl
        return jax.lax.pmean(value, DP_AXIS_NAME)

    actual = distributed(state, jax.device_put(batch, jax.P(None, DP_AXIS_NAME)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


@pytest.fixture
def trajectory():
    """Include natural termination, time truncation and a final bootstrap."""
    rewards = np.array(
        [[1, -2, 0.5], [0.5, 1, -1], [-1, 2, 0], [2, -0.5, 1]],
        dtype=np.float32,
    )
    values = np.arange(15, dtype=np.float32).reshape(5, 3) / 7 - 0.8
    dones = np.zeros_like(rewards)
    terminations = np.zeros_like(rewards)
    dones[1, 0] = 1  # Truncation stops the trace but retains one-step bootstrap.
    dones[2, 1] = terminations[2, 1] = 1
    return rewards, values, dones, terminations


@pytest.mark.parametrize("lambda_", [0.0, 0.75, 1.0])
@pytest.mark.parametrize("rho_clip,c_clip", [(1.0, 1.0), (0.7, 0.5), (2.0, 1.0)])
def test_vtrace_matches_numpy_backward_recurrence(
    trajectory, lambda_, rho_clip, c_clip, jit
):
    rewards, values, dones, terminations = trajectory
    rho = np.array(
        [[0.2, 1.2, 0.8], [1.5, 0.4, 2.5], [0.5, 1.0, 1.8], [2.0, 0.3, 1.1]],
        dtype=np.float32,
    )
    discount = 0.93
    expected = np.empty_like(rewards)
    correction = np.zeros(3, dtype=np.float32)
    for t in reversed(range(len(rewards))):
        td_error = np.minimum(rho[t], rho_clip) * (
            rewards[t] + discount * (1 - terminations[t]) * values[t + 1] - values[t]
        )
        correction = td_error + (
            discount
            * (1 - dones[t])
            * lambda_
            * np.minimum(rho[t], c_clip)
            * correction
        )
        expected[t] = values[t] + correction

    target = partial(
        compute_vtrace,
        discount=discount,
        lambda_=lambda_,
        clip_rho_threshold=rho_clip,
        clip_c_threshold=c_clip,
    )
    if jit:
        target = jax.jit(target)
    actual = target(
        *map(jnp.asarray, (rho, values[:-1], values[1:], rewards, dones, terminations))
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("lambda_", [0.0, 0.95, 1.0])
def test_onpolicy_vtrace_matches_gae_with_termination_and_truncation(
    trajectory, lambda_, jit
):
    rewards, values, dones, terminations = map(jnp.asarray, trajectory)
    target = partial(compute_vtrace, discount=0.99, lambda_=lambda_)
    if jit:
        target = jax.jit(target)
    actual = target(
        jnp.ones_like(rewards),
        values[:-1],
        values[1:],
        rewards,
        dones,
        terminations,
    )
    expected, advantages = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        terminations=terminations,
        gae_lambda=lambda_,
        discount=0.99,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual - values[:-1], advantages, rtol=1e-6, atol=1e-6)
