import jax
import jax.numpy as jnp
import chex
import numpy as np
import pytest

from evorl.utils.rl_toolkits import approximate_kl, compute_gae, compute_discount_return


def test_k3_weighted_expectation_matches_exact_categorical_kl(jit):
    behavior = np.array([0.2, 0.3, 0.5])
    current = np.array([0.4, 0.35, 0.25])
    logratio = np.log(current / behavior)
    estimate = (
        jax.jit(approximate_kl, static_argnames=("axis",)) if jit else approximate_kl
    )
    terms = estimate(jnp.asarray(logratio, jnp.float32), axis=())
    expected = np.sum(behavior * np.log(behavior / current))
    np.testing.assert_allclose(np.sum(behavior * terms), expected, rtol=1e-6)
    np.testing.assert_allclose(
        estimate(jnp.asarray(logratio, jnp.float32)), np.mean(terms), rtol=1e-6
    )
    assert np.all(np.asarray(terms) >= 0)


def test_k3_small_logratios_and_gradients_are_stable(jit):
    values = np.array([-1e-4, -1e-6, 0.0, 1e-6, 1e-4])
    expected = np.expm1(values) - values
    estimate = (
        jax.jit(lambda x: approximate_kl(x, axis=()))
        if jit
        else (lambda x: approximate_kl(x, axis=()))
    )
    actual = estimate(jnp.asarray(values, jnp.float32))
    np.testing.assert_allclose(actual, expected, rtol=0.1, atol=1e-13)
    gradient = jax.grad(lambda x: approximate_kl(x, axis=()).sum())
    if jit:
        gradient = jax.jit(gradient)
    np.testing.assert_allclose(
        gradient(jnp.asarray(values, jnp.float32)),
        np.expm1(values),
        rtol=0.1,
        atol=1e-7,
    )


@pytest.mark.parametrize("mode", ["k1", "k2"])
def test_alternative_kl_estimators_keep_their_existing_mean_api(mode):
    logratio = jnp.array([-0.2, 0.1, 0.3])
    terms = -logratio if mode == "k1" else 0.5 * logratio**2
    np.testing.assert_allclose(approximate_kl(logratio, mode), terms.mean())


def test_gae():
    T = 11
    B = 7
    keys = jax.random.split(jax.random.key(42), 3)
    rewards = jax.random.uniform(keys[0], (T, B), dtype=jnp.float32)
    values = jax.random.uniform(keys[1], (T + 1, B), dtype=jnp.float32)

    dones = jnp.zeros((T, B))
    term_steps = [T - i for i in range(1, B + 1)]

    for i in range(B):
        dones = dones.at[term_steps[i] - 1 :, i].set(1)

    compute_gae(rewards, values, dones, dones, 0.95, 0.99)


def test_discount_return():
    T = 1000
    B = 3
    rewards = jnp.arange(1, T * B + 1, dtype=jnp.float32).reshape(T, B)
    dones = jnp.zeros_like(rewards)
    dones = dones.at[-1].set(1)

    def _real_discount_return(rewards, discount):
        discount_return = jnp.zeros_like(rewards[0])
        for i in range(rewards.shape[0]):
            discount_return += rewards[i] * jnp.power(discount, i)
        return discount_return

    discount = 0.9
    discount_return = compute_discount_return(rewards, dones, discount)
    discount_return_real = _real_discount_return(rewards, discount)
    chex.assert_trees_all_close(discount_return, discount_return_real)

    discount = 1.0
    discount_return = compute_discount_return(rewards, dones, discount)
    discount_return_real = _real_discount_return(rewards, discount)
    chex.assert_trees_all_close(discount_return, discount_return_real)


def test_discount_return_with_dones():
    T = 7
    B = 4
    rewards = jnp.arange(1, T * B + 1, dtype=jnp.float32).reshape(T, B)
    dones = jnp.zeros_like(rewards)

    term_steps = [T - i for i in range(1, B + 1)]

    for i in range(B):
        dones = dones.at[term_steps[i] - 1 :, i].set(1)

    def _real_discount_return(rewards, discount, term_steps):
        discount_return = jnp.zeros_like(rewards[0])
        for i in range(rewards.shape[1]):
            reward_i = rewards[: term_steps[i], i]
            for t in range(term_steps[i]):
                discount_return = discount_return.at[i].add(
                    reward_i[t] * jnp.power(discount, t)
                )
        return discount_return

    discount = 0.9
    discount_return = compute_discount_return(rewards, dones, discount)
    discount_return_real = _real_discount_return(rewards, discount, term_steps)
    chex.assert_trees_all_close(discount_return, discount_return_real)

    discount = 1.0
    discount_return = compute_discount_return(rewards, dones, discount)
    discount_return_real = _real_discount_return(rewards, discount, term_steps)
    chex.assert_trees_all_close(discount_return, discount_return_real)
