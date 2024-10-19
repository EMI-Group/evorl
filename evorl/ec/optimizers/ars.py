import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.types import (
    PyTreeData,
    Params,
    pytree_field,
)
from evorl.utils.jax_utils import (
    rng_split_like_tree,
)

from .utils import weight_sum
from .ec_optimizer import EvoOptimizer, ECState


class ARSState(PyTreeData):
    mean: chex.ArrayTree
    opt_state: optax.OptState
    key: chex.PRNGKey


class ARS(EvoOptimizer):
    pop_size: int
    num_elites: int
    lr: float
    noise_std: float
    fitness_std_eps: float = 1e-8

    optimizer: optax.GradientTransformation = pytree_field(
        pytree_node=False, lazy_init=True
    )

    def __post_init__(self):
        assert (
            self.pop_size > 0 and self.pop_size % 2 == 0
        ), "pop_size must be positive even number"

        optimizer = optax.adam(learning_rate=self.lr)
        self.set_frozen_attr("optimizer", optimizer)

    def init(self, mean: Params, key: chex.PRNGKey) -> ARSState:
        opt_state = self.optimizer.init(mean)
        return ARSState(mean=mean, opt_state=opt_state, key=key)

    def ask(self, state: ARSState) -> tuple[Params, ECState]:
        key, sample_key = jax.random.split(state.key)
        sample_keys = rng_split_like_tree(sample_key, state.mean)

        noise = jtu.tree_map(
            lambda x, k: jax.random.normal(k, shape=(self.pop_size // 2, *x.shape)),
            state.mean,
            sample_keys,
        )
        noise = jtu.tree_map(lambda z: jnp.concatenate([z, -z], axis=0), noise)

        pop = jtu.tree_map(
            lambda m, z: m + self.noise_std * z,
            state.mean,
            noise,
        )
        return pop, state.replace(key=key)

    def tell(self, state: ARSState, xs: Params, fitnesses: chex.Array) -> ARSState:
        half_pop_size = self.pop_size // 2

        noise = jtu.tree_map(
            lambda x, m: (x[:half_pop_size] - m) / self.noise_std,
            xs,
            state.mean,
        )

        fit_p = fitnesses[:half_pop_size]  # r_positive
        fit_n = fitnesses[half_pop_size:]  # r_negtive
        elites_indices = jax.lax.top_k(jnp.maximum(fit_p, fit_n), self.num_elites)[1]

        fitnesses_elite = jnp.concatenate(
            [fit_p[elites_indices], fit_n[elites_indices]]
        )
        # Add small constant to ensure non-zero division stability
        fitness_std = jnp.std(fitnesses_elite) + self.fitness_std_eps

        fit_diff = (fit_p[elites_indices] - fit_n[elites_indices]) / fitness_std

        grad = jtu.tree_map(
            # Note: we need additional "-1.0" since we are maximizing the fitness
            lambda z: (-weight_sum(z[elites_indices], fit_diff) / (self.num_elites)),
            noise,
        )

        update, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, update)

        return state.replace(mean=mean, opt_state=opt_state)
