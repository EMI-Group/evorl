import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.types import PyTreeData, Params, PyTreeDict, pytree_field
from evorl.utils.jax_utils import rng_split_like_tree

from .utils import weight_sum, ExponentialScheduleSpec
from .ec_optimizer import EvoOptimizer


class VanillaESState(PyTreeData):
    """State of the VanillaES."""

    mean: chex.ArrayTree
    noise_std: chex.Array
    key: chex.PRNGKey
    noise: None | chex.ArrayTree = None


class VanillaES(EvoOptimizer):
    """Canonical Evolution Strategies.

    Paper: [Back to basics: Benchmarking canonical evolution strategies for playing atari](https://arxiv.org/abs/1802.08842)
    """

    pop_size: int
    num_elites: int
    noise_std_schedule: ExponentialScheduleSpec
    elite_weights: chex.Array = pytree_field(init=False)

    def __post_init__(self):
        elite_weights = jnp.log(self.num_elites + 0.5) - jnp.log(
            jnp.arange(1, self.num_elites + 1)
        )
        self.elite_weights = elite_weights / elite_weights.sum()

    def init(self, mean: Params, key: chex.PRNGKey) -> VanillaESState:
        return VanillaESState(
            mean=mean,
            noise_std=jnp.float32(self.noise_std_schedule.init),
            key=key,
        )

    def ask(self, state: VanillaESState) -> tuple[Params, VanillaESState]:
        key, sample_key = jax.random.split(state.key)
        sample_keys = rng_split_like_tree(sample_key, state.mean)

        noise = jtu.tree_map(
            lambda x, k: jax.random.normal(k, shape=(self.pop_size, *x.shape))
            * state.noise_std,
            state.mean,
            sample_keys,
        )

        pop = jtu.tree_map(
            lambda m, z: m + z,
            state.mean,
            noise,
        )
        return pop, state.replace(key=key, noise=noise)

    def tell(
        self, state: VanillaESState, fitnesses: chex.Array
    ) -> tuple[PyTreeDict, VanillaESState]:
        elite_indices = jax.lax.top_k(fitnesses, self.num_elites)[1]

        mean = jtu.tree_map(
            lambda x, z: x + weight_sum(z[elite_indices], self.elite_weights),
            state.mean,
            state.noise,
        )

        noise_std = optax.incremental_update(
            self.noise_std_schedule.final,
            state.noise_std,
            1 - self.noise_std_schedule.decay,
        )

        return PyTreeDict(), state.replace(mean=mean, noise_std=noise_std, noise=None)


class VanillaESMod(VanillaES):
    """Variant of VanillaES.

    Add `external_size` number of external individuals and corresponding fitnesses during the ES update by `tell_external()`

    Attributes:
        external_size: number of external individuals
        mix_strategy: strategy to mix external individuals with the elites.
            - "always": always mix external individuals with elites
            - "normal": concat external individuals to the population and select `num_elites` elites from the combined population.
    """

    external_size: int
    mix_strategy: str = "always"

    def __post_init__(self):
        super().__post_init__()
        assert self.num_elites >= self.external_size
        assert self.mix_strategy in ["always", "normal"]

    def tell_external(
        self, state: VanillaESState, fitnesses: chex.Array
    ) -> tuple[PyTreeDict, VanillaESState]:
        chex.assert_shape(fitnesses, (self.pop_size + self.external_size,))
        chex.assert_tree_shape_prefix(
            state.noise, (self.pop_size + self.external_size,)
        )

        if self.mix_strategy == "always":
            # select (self.num_elites-self.external_size) elites from pop
            # then insert all external individuals and sort them.

            # Note: user should ensure external individuals and fitnesses are concated behind the pop.
            # TODO: need to improve
            pop_fitnesses = fitnesses[: self.pop_size]
            external_fitnesses = fitnesses[self.pop_size :]

            pop_elite_fitnesses, pop_elite_indices = jax.lax.top_k(
                pop_fitnesses, self.num_elites - self.external_size
            )

            elite_fitnesses = jnp.concatenate([pop_elite_fitnesses, external_fitnesses])
            elite_indices = jnp.concatenate(
                [
                    pop_elite_indices,
                    jnp.arange(
                        self.pop_size,
                        self.pop_size + self.external_size,
                        dtype=jnp.int32,
                    ),
                ]
            )

            elite_indices = elite_indices[jnp.argsort(elite_fitnesses, descending=True)]
        else:
            elite_indices = jax.lax.top_k(fitnesses, self.num_elites)[1]

        mean = jtu.tree_map(
            lambda x, z: x + weight_sum(z[elite_indices], self.elite_weights),
            state.mean,
            state.noise,
        )

        noise_std = optax.incremental_update(
            self.noise_std_schedule.final,
            state.noise_std,
            1 - self.noise_std_schedule.decay,
        )

        return PyTreeDict(), state.replace(mean=mean, noise_std=noise_std, noise=None)
