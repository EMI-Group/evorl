# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: Simple random search provides a competitive approach to reinforcement learning(ARS)
# Link: https://arxiv.org/pdf/1803.07055.pdf
#
# 2. This code has been inspired by or utilizes the algorithmic implementation from evosax.
# More information about evosax can be found at the following URL:
# GitHub Link: https://github.com/RobertTLange/evosax
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import optax

from evox import Algorithm, State, use_state, utils


class ARS(Algorithm):
    def __init__(
        self,
        center_init: jax.Array,
        pop_size: int,
        num_elites: int,
        optimizer: str = "adam",
        lr: float = 0.05,
        sigma: float = 0.03,
        eps: float = 1e-08,
    ):
        super().__init__()

        assert (
            pop_size > 0 and pop_size % 2 == 0
        ), "pop_size must be a positive even number"

        self.dim = center_init.shape[0]
        self.center_init = center_init
        self.pop_size = pop_size
        self.num_elites = num_elites
        self.lr = lr
        self.sigma = sigma
        self.eps = eps

        if optimizer == "adam":
            self.optimizer = utils.OptaxWrapper(
                optax.adam(learning_rate=lr), center_init
            )
        else:
            self.optimizer = None

    def setup(self, key):
        return State(
            key=key,
            center=self.center_init,
            delta=jax.random.normal(key, (self.pop_size // 2, self.dim)),
        )

    def ask(self, state):
        key, sample_key = jax.random.split(state.key)
        z_plus = jax.random.normal(sample_key, (self.pop_size // 2, self.dim))
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state.center + self.sigma * z
        return x, state.replace(key=key, delta=z_plus)

    def tell(self, state, fitness):
        half_pop_size = self.pop_size // 2

        fit_1 = fitness[:half_pop_size]
        fit_2 = fitness[half_pop_size:]
        # elite_idx = jnp.minimum(fit_1, fit_2).argsort()[: self.num_elites]
        elite_idx = jax.lax.top_k(-jnp.minimum(fit_1, fit_2), self.num_elites)[1]

        fitness_elite = jnp.concatenate([fit_1[elite_idx], fit_2[elite_idx]])
        # Add small constant to ensure non-zero division stability
        fitness_std = jnp.std(fitness_elite) + self.eps

        fit_diff = fit_1[elite_idx] - fit_2[elite_idx]

        grad = (state.delta[elite_idx] @ fit_diff) / (self.num_elites * fitness_std)

        if self.optimizer is None:
            center = state.center - self.lr * grad
        else:
            updates, state = use_state(self.optimizer.update)(state, grad)
            center = optax.apply_updates(state.center, updates)

        return state.replace(center=center)
