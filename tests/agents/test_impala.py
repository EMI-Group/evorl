"""V-trace targets against independent recurrences and on-policy GAE."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.algorithms.impala import compute_vtrace
from evorl.utils.rl_toolkits import compute_gae


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
