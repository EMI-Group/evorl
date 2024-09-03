from abc import ABCMeta, abstractmethod

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.types import (
    PyTreeData,
    PyTreeNode,
    Params,
    pytree_field,
)
from evorl.utils.jax_utils import (
    rng_split_like_tree,
)


class EvolutionOptimizer(PyTreeNode, metaclass=ABCMeta):
    @abstractmethod
    def init(self, *args, **kwargs) -> chex.ArrayTree:
        pass

    @abstractmethod
    def update(
        self, state: chex.ArrayTree, offsprings: chex.ArrayTree, fitness: chex.Array
    ) -> chex.ArrayTree:
        pass

    @abstractmethod
    def sample(
        self, state: chex.ArrayTree, pop_size: int, key: chex.PRNGKey
    ) -> chex.ArrayTree:
        pass


class DiagCEMState(PyTreeData):
    mean: chex.ArrayTree
    variance: chex.ArrayTree
    cov_noise: chex.ArrayTree


class DiagCEM(EvolutionOptimizer):
    num_elites: int  # number of good offspring to update the pop
    init_diagonal_variance: float = 1e-2
    final_diagonal_variance: float = 1e-5
    diagonal_variance_decay: float = 0.05
    weighted_update: bool = True
    rank_weight_shift: float = 1.0
    mirror_sampling: bool = False
    elite_weights: chex.Array = pytree_field(lazy_init=True)

    def __post_init__(self):
        if self.weighted_update:
            # this logarithmic rank-based weighting is from CEM-RL
            elite_weights = jnp.log(
                (self.num_elites + self.rank_weight_shift)
                / jnp.arange(1, self.num_elites + 1)
            )
        else:
            elite_weights = jnp.ones((self.num_elites,))

        # elite_weights = elite_weights / elite_weights.sum()

        self.set_frozen_attr("elite_weights", elite_weights)

    def init(self, init_actor_params: Params) -> DiagCEMState:
        variance = jtu.tree_map(
            lambda x: jnp.full_like(x, self.init_diagonal_variance), init_actor_params
        )

        return DiagCEMState(
            mean=init_actor_params,
            variance=variance,
            cov_noise=jnp.float32(self.init_diagonal_variance),
        )

    def update(
        self, state: DiagCEMState, offsprings: chex.ArrayTree, fitnesses: chex.Array
    ) -> DiagCEMState:
        # fitness: episode_return, higher is better
        elites_indices = jax.lax.top_k(fitnesses, self.num_elites)[1]

        cov_noise = optax.incremental_update(
            self.final_diagonal_variance, state.cov_noise, self.diagonal_variance_decay
        )

        mean = jtu.tree_map(
            lambda x: jnp.average(
                x[elites_indices], axis=0, weights=self.elite_weights
            ),
            offsprings,
        )

        def var_update(mean, x):
            t1 = jnp.square(x[elites_indices] - mean)
            # TODO: do we need extra division by num_elites mentioned CEM-RL?
            t2 = jnp.average(t1, axis=0, weights=self.elite_weights) + cov_noise
            return t2

        variance = jtu.tree_map(
            var_update,
            state.mean,  # old mean
            offsprings,
        )

        return state.replace(mean=mean, variance=variance, cov_noise=cov_noise)

    def sample(
        self, state: DiagCEMState, pop_size: int, key: chex.PRNGKey
    ) -> chex.ArrayTree:
        keys = rng_split_like_tree(key, state.mean)

        if self.mirror_sampling:
            assert (
                pop_size > 0 and pop_size % 2 == 0
            ), "pop_size must be even for mirror sampling"
            half_noise = jtu.tree_map(
                lambda x, var, k: jax.random.normal(k, (pop_size // 2, *x.shape))
                * jnp.sqrt(var),
                state.mean,
                state.variance,
                keys,
            )

            noise = jtu.tree_map(
                lambda x: jnp.concatenate([x, -x], axis=0),
                half_noise,
            )

        else:
            noise = jtu.tree_map(
                lambda x, var, k: jax.random.normal(k, (pop_size, *x.shape))
                * jnp.sqrt(var),
                state.mean,
                state.variance,
                keys,
            )

        # noise: (#pop, ...)
        # mean: (...)

        return jtu.tree_map(lambda mean, noise: mean + noise, state.mean, noise)
