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

from .utils import ExponentialScheduleSpec, weight_sum
from .ec_optimizer import EvoOptimizer, ECState


class SepCEMState(PyTreeData):
    mean: chex.ArrayTree
    variance: chex.ArrayTree
    cov_noise: chex.ArrayTree
    key: chex.PRNGKey


class SepCEM(EvoOptimizer):
    pop_size: int
    num_elites: int  # number of good offspring to update the pop
    diagonal_variance: ExponentialScheduleSpec

    weighted_update: bool = True
    rank_weight_shift: float = 1.0
    mirror_sampling: bool = False
    elite_weights: chex.Array = pytree_field(lazy_init=True)

    def __post_init__(self):
        assert self.pop_size > 0, "pop_size must be positive"
        if self.mirror_sampling:
            assert self.pop_size % 2 == 0, "pop_size must be even for mirror sampling"

        if self.weighted_update:
            elite_weights = jnp.log(self.num_elites + self.rank_weight_shift) - jnp.log(
                jnp.arange(1, self.num_elites + 1)
            )
        else:
            elite_weights = jnp.ones((self.num_elites,))

        elite_weights = elite_weights / elite_weights.sum()

        self.set_frozen_attr("elite_weights", elite_weights)

    def init(self, mean: Params, key: chex.PRNGKey) -> SepCEMState:
        variance = jtu.tree_map(
            lambda x: jnp.full_like(x, self.diagonal_variance.init), mean
        )

        return SepCEMState(
            mean=mean,
            variance=variance,
            cov_noise=jnp.float32(self.diagonal_variance.init),
            key=key,
        )

    def tell(
        self, state: SepCEMState, xs: chex.ArrayTree, fitnesses: chex.Array
    ) -> SepCEMState:
        # fitness: episode_return, higher is better
        elites_indices = jax.lax.top_k(fitnesses, self.num_elites)[1]

        cov_noise = optax.incremental_update(
            self.diagonal_variance.final, state.cov_noise, self.diagonal_variance.decay
        )

        mean = jtu.tree_map(
            lambda x: weight_sum(x[elites_indices], self.elite_weights),
            xs,
        )

        def var_update(m, x):
            x_norm = jnp.square(x[elites_indices] - m)
            # TODO: do we need extra division by num_elites mentioned in CEM-RL?
            return weight_sum(x_norm, self.elite_weights) + cov_noise

        variance = jtu.tree_map(
            var_update,
            state.mean,  # old mean
            xs,
        )

        return state.replace(mean=mean, variance=variance, cov_noise=cov_noise)

    def ask(self, state: SepCEMState) -> tuple[chex.ArrayTree, ECState]:
        key, sample_key = jax.random.split(state.key)
        sample_keys = rng_split_like_tree(sample_key, state.mean)

        if self.mirror_sampling:
            half_noise = jtu.tree_map(
                lambda x, var, k: jax.random.normal(k, (self.pop_size // 2, *x.shape))
                * jnp.sqrt(var),
                state.mean,
                state.variance,
                sample_keys,
            )

            noise = jtu.tree_map(
                lambda x: jnp.concatenate([x, -x], axis=0),
                half_noise,
            )

        else:
            noise = jtu.tree_map(
                lambda x, var, k: jax.random.normal(k, (self.pop_size, *x.shape))
                * jnp.sqrt(var),
                state.mean,
                state.variance,
                sample_keys,
            )

        # noise: (#pop, ...)
        # mean: (...)

        pop = jtu.tree_map(lambda mean, noise: mean + noise, state.mean, noise)
        state = state.replace(key=key)

        return pop, state
