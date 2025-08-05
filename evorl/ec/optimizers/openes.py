from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.types import PyTreeData, pytree_field, Params, PyTreeDict
from evorl.utils.jax_utils import rng_split_like_tree, invert_permutation
from evorl.utils.ec_utils import ParamVectorSpec

from .utils import ExponentialScheduleSpec, weight_sum, optimizer_map
from .ec_optimizer import EvoOptimizer, ECState


def compute_ranks(x):
    """Get ranks in [0, len(x)-1].

    This is different from `scipy.stats.rankdata`, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = invert_permutation(jnp.argsort(x))
    return ranks


def compute_centered_ranks(x):
    """Get centered ranks in [-0.5, 0.5]."""
    y = compute_ranks(x)
    y /= x.size - 1
    y -= 0.5
    return y


class OpenESState(PyTreeData):
    """State of the OpenES."""

    mean: chex.ArrayTree
    opt_state: optax.OptState
    noise_std: chex.Array
    key: chex.PRNGKey
    noise: None | chex.ArrayTree = None


class OpenES(EvoOptimizer):
    """OpenAI ES."""

    pop_size: int
    lr_schedule: ExponentialScheduleSpec
    noise_std_schedule: ExponentialScheduleSpec
    mirror_sampling: bool = True
    optimizer_name: str = "adam"
    weight_decay: float | None = None

    fitness_shaping_fn: Callable[[chex.Array], chex.Array] = pytree_field(
        static=True, default=compute_centered_ranks
    )
    optimizer: optax.GradientTransformation = pytree_field(static=True, init=False)

    def __post_init__(self):
        assert self.pop_size > 0, "pop_size must be positive"
        if self.mirror_sampling:
            assert self.pop_size % 2 == 0, "pop_size must be even for mirror sampling"

        self.optimizer = optax.inject_hyperparams(optimizer_map[self.optimizer_name])(
            learning_rate=self.lr_schedule.init
        )

    def init(self, mean: Params, key: chex.PRNGKey) -> ECState:
        return OpenESState(
            mean=mean,
            opt_state=self.optimizer.init(mean),
            noise_std=jnp.float32(self.noise_std_schedule.init),
            key=key,
        )

    def ask(self, state: ECState) -> tuple[chex.ArrayTree, ECState]:
        """Generate new candidate solutions."""
        key, sample_key = jax.random.split(state.key)
        sample_keys = rng_split_like_tree(sample_key, state.mean)

        if self.mirror_sampling:
            noise = jtu.tree_map(
                lambda x, k: jax.random.normal(k, shape=(self.pop_size // 2, *x.shape)),
                state.mean,
                sample_keys,
            )
            noise = jtu.tree_map(lambda z: jnp.concatenate([z, -z], axis=0), noise)
        else:
            noise = jtu.tree_map(
                lambda x, k: jax.random.normal(k, shape=(self.pop_size, *x.shape)),
                state.mean,
                sample_keys,
            )

        pop = jtu.tree_map(
            lambda m, z: m + state.noise_std * z,
            state.mean,
            noise,
        )
        state = state.replace(key=key, noise=noise)

        return pop, state

    def tell(
        self, state: ECState, fitnesses: chex.Array
    ) -> tuple[PyTreeDict, OpenESState]:
        """Update the optimizer state based on the fitnesses of the candidate solutions."""
        transformed_fitnesses = self.fitness_shaping_fn(fitnesses)

        # grad = 1/(N*sigma^2) * sum(F_i*(x_i-m))
        grad = jtu.tree_map(
            # Note: we need additional "-1.0" since we are maximizing the fitness
            lambda z: (
                -weight_sum(z, transformed_fitnesses)
                / (self.pop_size * state.noise_std)
            ),
            state.noise,
        )

        # add L2 weight decay
        if self.weight_decay is not None:
            grad = jtu.tree_map(
                lambda g, x: g + self.weight_decay * x,
                grad,
                state.mean,
            )

        update, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, update)

        opt_state.hyperparams["learning_rate"] = optax.incremental_update(
            self.lr_schedule.final,
            opt_state.hyperparams["learning_rate"],
            1 - self.lr_schedule.decay,
        )

        noise_std = optax.incremental_update(
            self.noise_std_schedule.final,
            state.noise_std,
            1 - self.noise_std_schedule.decay,
        )

        return PyTreeDict(), state.replace(
            mean=mean, opt_state=opt_state, noise_std=noise_std, noise=None
        )


class OpenESNoiseTableState(PyTreeData):
    """State of the OpenES with noise table."""

    mean: chex.ArrayTree
    opt_state: optax.OptState
    noise_std: chex.Array
    noise_table: chex.ArrayTree
    key: chex.PRNGKey
    noise: None | chex.ArrayTree = None


class OpenESNoiseTable(EvoOptimizer):
    """OpenAI ES with noise table."""

    pop_size: int
    noise_table_size: int
    lr_schedule: ExponentialScheduleSpec
    noise_std_schedule: ExponentialScheduleSpec
    mirror_sampling: bool = True
    optimizer_name: str = "adam"
    weight_decay: float | None = None

    fitness_shaping_fn: Callable[[chex.Array], chex.Array] = pytree_field(
        static=True, default=compute_centered_ranks
    )
    optimizer: optax.GradientTransformation = pytree_field(static=True, init=False)

    def __post_init__(self):
        assert self.pop_size > 0, "pop_size must be positive"
        if self.mirror_sampling:
            assert self.pop_size % 2 == 0, "pop_size must be even for mirror sampling"

        self.optimizer = optax.inject_hyperparams(optimizer_map[self.optimizer_name])(
            learning_rate=self.lr_schedule.init
        )

    def init(self, mean: Params, key: chex.PRNGKey) -> ECState:
        key, noise_table_key = jax.random.split(key)
        noise_table = jax.random.normal(noise_table_key, shape=(self.noise_table_size,))

        return OpenESNoiseTableState(
            mean=mean,
            opt_state=self.optimizer.init(mean),
            noise_std=jnp.float32(self.noise_std_schedule.init),
            noise_table=noise_table,
            key=key,
        )

    def ask(self, state: ECState) -> tuple[chex.ArrayTree, ECState]:
        """Generate new candidate solutions."""
        key, sample_key = jax.random.split(state.key)
        # sample_keys = rng_split_like_tree(sample_key, state.mean)

        param_vec_spec = ParamVectorSpec(state.mean)

        def sample_from_noise_table(idx):
            return jax.lax.dynamic_slice_in_dim(
                state.noise_table, idx, param_vec_spec.vec_size, axis=0
            )

        if self.mirror_sampling:
            noise_idx = jax.random.randint(
                sample_key,
                shape=(self.pop_size // 2,),
                minval=0,
                maxval=self.noise_table_size - param_vec_spec.vec_size,
            )
            noise = param_vec_spec.to_tree(jax.vmap(sample_from_noise_table)(noise_idx))

            noise = jtu.tree_map(lambda z: jnp.concatenate([z, -z], axis=0), noise)
        else:
            noise_idx = jax.random.randint(
                sample_key,
                shape=(self.pop_size,),
                minval=0,
                maxval=self.noise_table_size - param_vec_spec.vec_size,
            )
            noise = param_vec_spec.to_tree(jax.vmap(sample_from_noise_table)(noise_idx))

        pop = jtu.tree_map(
            lambda m, z: m + state.noise_std * z,
            state.mean,
            noise,
        )
        state = state.replace(key=key, noise=noise)

        return pop, state

    def tell(
        self, state: ECState, fitnesses: chex.Array
    ) -> tuple[PyTreeDict, OpenESState]:
        """Update the optimizer state based on the fitnesses of the candidate solutions."""
        transformed_fitnesses = self.fitness_shaping_fn(fitnesses)

        # grad = 1/(N*sigma^2) * sum(F_i*(x_i-m))
        grad = jtu.tree_map(
            # Note: we need additional "-1.0" since we are maximizing the fitness
            lambda z: (
                -weight_sum(z, transformed_fitnesses)
                / (self.pop_size * state.noise_std)
            ),
            state.noise,
        )

        # add L2 weight decay
        if self.weight_decay is not None:
            grad = jtu.tree_map(
                lambda g, x: g + self.weight_decay * x,
                grad,
                state.mean,
            )

        update, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, update)

        opt_state.hyperparams["learning_rate"] = optax.incremental_update(
            self.lr_schedule.final,
            opt_state.hyperparams["learning_rate"],
            1 - self.lr_schedule.decay,
        )

        noise_std = optax.incremental_update(
            self.noise_std_schedule.final,
            state.noise_std,
            1 - self.noise_std_schedule.decay,
        )

        return PyTreeDict(), state.replace(
            mean=mean, opt_state=opt_state, noise_std=noise_std, noise=None
        )
