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

from .utils import weight_sum, ExponentialScheduleSpec
from .ec_optimizer import EvoOptimizer, ECState


class VanillaESState(PyTreeData):
    mean: chex.ArrayTree
    noise_std: chex.Array
    key: chex.PRNGKey


class VanillaES(EvoOptimizer):
    pop_size: int
    num_elites: int
    noise_std_schedule: ExponentialScheduleSpec
    elite_weights: chex.Array = pytree_field(lazy_init=True)

    def __post_init__(self):
        elite_weights = jnp.log(self.num_elites + 0.5) - jnp.log(
            jnp.arange(1, self.num_elites + 1)
        )
        elite_weights = elite_weights / elite_weights.sum()
        self.set_frozen_attr("elite_weights", elite_weights)

    def init(self, mean: Params, key: chex.PRNGKey) -> VanillaESState:
        return VanillaESState(
            mean=mean,
            noise_std=jnp.float32(self.noise_std_schedule.init),
            key=key,
        )

    def ask(self, state: VanillaESState) -> tuple[Params, ECState]:
        key, sample_key = jax.random.split(state.key)
        sample_keys = rng_split_like_tree(sample_key, state.mean)

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
        return pop, state.replace(key=key)

    def tell(
        self, state: VanillaESState, xs: Params, fitnesses: chex.Array
    ) -> VanillaESState:
        elites_indices = jax.lax.top_k(fitnesses, self.num_elites)[1]

        noise = jtu.tree_map(lambda x, m: (x - m), xs, state.mean)

        mean = jtu.tree_map(
            lambda x, z: x + weight_sum(z[elites_indices], self.elite_weights),
            state.mean,
            noise,
        )

        noise_std = optax.incremental_update(
            self.noise_std_schedule.final,
            state.noise_std,
            1 - self.noise_std_schedule.decay,
        )

        return state.replace(mean=mean, noise_std=noise_std)
