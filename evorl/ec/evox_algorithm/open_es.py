# --------------------------------------------------------------------------------------
# This code implements algorithms described in the following papers:
#
# Title: Exponential Natural Evolution Strategies (XNES)
# Link: https://dl.acm.org/doi/abs/10.1145/1830483.1830557
#
# Title: Natural Evolution Strategies (SeparableNES)
# Link: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import optax

from evox import Algorithm, State, use_state, utils
from evorl.utils.jax_utils import invert_permutation


def compute_ranks(x):
    """Returns ranks in [0, len(x)-1].

    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = invert_permutation(jnp.argsort(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x)
    y /= x.size - 1
    y -= 0.5
    return y


class OpenES(Algorithm):
    """OpenAI ES.

    Paper: [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864).
    """

    def __init__(
        self,
        center_init,
        pop_size,
        learning_rate,
        noise_std,
        optimizer=None,
        mirror_sampling=True,
    ):
        assert noise_std > 0
        assert learning_rate > 0
        assert pop_size > 0

        if mirror_sampling is True:
            assert pop_size % 2 == 0, (
                "When mirrored_sampling is True, pop_size must be a multiple of 2."
            )

        self.dim = center_init.shape[0]
        self.center_init = center_init
        self.pop_size = pop_size
        self.learning_rate = learning_rate
        self.noise_std = noise_std
        self.mirror_sampling = mirror_sampling

        if optimizer == "adam":
            self.optimizer = utils.OptaxWrapper(
                optax.adam(learning_rate=learning_rate), center_init
            )
        else:
            self.optimizer = None

    def setup(self, key):
        # population = jnp.tile(self.center_init, (self.pop_size, 1))
        noise = jax.random.normal(key, shape=(self.pop_size, self.dim))
        return State(center=self.center_init, noise=noise, key=key)

    def ask(self, state):
        key, noise_key = jax.random.split(state.key)
        if self.mirror_sampling:
            noise = jax.random.normal(noise_key, shape=(self.pop_size // 2, self.dim))
            noise = jnp.concatenate([noise, -noise], axis=0)
        else:
            noise = jax.random.normal(noise_key, shape=(self.pop_size, self.dim))
        population = state.center[jnp.newaxis, :] + self.noise_std * noise

        return population, state.replace(key=key, noise=noise)

    def tell(self, state, fitness):
        # Tips: by default, Algorithm handle minimization problem
        weights = -compute_centered_ranks(-fitness)
        grad = state.noise.T @ weights / (self.pop_size * self.noise_std)

        if self.optimizer is None:
            center = state.center - self.learning_rate * grad
        else:
            updates, state = use_state(self.optimizer.update)(state, grad)
            center = optax.apply_updates(state.center, updates)

        return state.replace(center=center)
