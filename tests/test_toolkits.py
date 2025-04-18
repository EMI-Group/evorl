import jax
import jax.numpy as jnp
import chex

from evorl.utils.rl_toolkits import compute_gae, compute_discount_return


def test_gae():
    T = 11
    B = 7
    keys = jax.random.split(jax.random.PRNGKey(42), 3)
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
