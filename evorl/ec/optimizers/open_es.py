from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.types import PyTreeData, pytree_field, Params
from evorl.utils.jax_utils import rng_split_like_tree

from .ec_optimizer import EvoOptimizer, ECState


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = jnp.arange(len(x))[x.argsort(descending=True)]
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x)
    y /= x.size - 1
    y -= 0.5
    return y


class ScheduleSpec(PyTreeData):
    init: float
    final: float
    decay: float


class OpenESState(PyTreeData):
    mean: chex.ArrayTree
    opt_state: optax.OptState
    noise_stdev: float


class OpenES(EvoOptimizer):
    pop_size: int
    lr_schedule: ScheduleSpec
    noise_stdev_schedule: ScheduleSpec
    mirror_sampling: bool = True
    fitness_shaping_fn: Callable[[chex.Array], chex.Array] = pytree_field(
        pytree_node=False, default=compute_centered_ranks
    )
    optimizer: optax.GradientTransformation = pytree_field(
        pytree_node=False, lazy_init=True
    )

    def __post_init__(self):
        assert self.pop_size > 0, "pop_size must be positive"
        if self.mirror_sampling:
            assert self.pop_size % 2 == 0, "pop_size must be even for mirror sampling"

        optimizer = optax.inject_hyperparams(
            optax.adam, static_args=("b1", "b2", "eps", "eps_root")
        )(learning_rate=self.lr_schedule.init)
        self.set_frozen_attr("optimizer", optimizer)

    def init(self, mean: Params) -> ECState:
        return OpenESState(
            mean=mean,
            opt_state=self.optimizer.init(mean),
            noise_stdev=self.noise_stdev_schedule.init,
        )

    def tell(
        self, state: ECState, xs: chex.ArrayTree, fitnesses: chex.Array
    ) -> ECState:
        "Update the optimizer state based on the fitnesses of the candidate solutions"

        transformed_fitnesses = self.fitness_shaping_fn(fitnesses)

        opt_state = state.opt_state

        # [pop_size, ...]
        noise = jtu.tree_map(lambda x, m: x - m, xs, state.mean)
        grad = jtu.tree_map(
            lambda n: jnp.average(n, axis=0, weights=transformed_fitnesses)
            / (self.pop_size * state.noise_stdev),
            noise,
        )
        update, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, update)

        opt_state.hyperparams["learning_rate"] = optax.incremental_update(
            self.lr_schedule.final,
            opt_state.hyperparams["learning_rate"],
            1 - self.lr_schedule.decay,
        )

        noise_stdev = optax.incremental_update(
            self.noise_stdev_schedule.final,
            state.noise_stdev,
            1 - self.noise_stdev_schedule.decay,
        )

        return state.replace(mean=mean, opt_state=opt_state, noise_stdev=noise_stdev)

    def ask(self, state: ECState, key: chex.PRNGKey) -> chex.ArrayTree:
        "Generate new candidate solutions"
        keys = rng_split_like_tree(key, state.mean)

        if self.mirror_sampling:
            noise = jtu.tree_map(
                lambda x, k: jax.random.normal(k, shape=(self.pop_size // 2, *x.shape)),
                state.mean,
                keys,
            )
            noise = jtu.tree_map(lambda x: jnp.concatenate([x, -x], axis=0), noise)
        else:
            noise = jtu.tree_map(
                lambda x, k: jax.random.normal(k, shape=(self.pop_size, *x.shape)),
                state.mean,
                keys,
            )

        return jtu.tree_map(lambda mean, noise: mean + noise, state.mean, noise)
