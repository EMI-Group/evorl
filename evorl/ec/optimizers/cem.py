import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.types import PyTreeData, PyTreeDict, Params, pytree_field
from evorl.utils.jax_utils import rng_split_like_tree

from .utils import ExponentialScheduleSpec, weight_sum
from .ec_optimizer import EvoOptimizer, ECState


class SepCEMState(PyTreeData):
    """State of the SepCEM."""

    mean: chex.ArrayTree
    variance: chex.ArrayTree
    cov_eps: chex.ArrayTree
    key: chex.PRNGKey
    pop: None | chex.ArrayTree = None


class SepCEM(EvoOptimizer):
    """Sep Cross-Entropy Method."""

    pop_size: int
    num_elites: int  # number of good offspring to update the pop
    cov_eps_schedule: ExponentialScheduleSpec

    weighted_update: bool = True
    rank_weight_shift: float = 1.0
    mirror_sampling: bool = False
    elite_weights: chex.Array = pytree_field(init=False)

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

        self.elite_weights = elite_weights / elite_weights.sum()

    def init(self, mean: Params, key: chex.PRNGKey) -> SepCEMState:
        variance = jtu.tree_map(
            lambda x: jnp.full_like(x, self.cov_eps_schedule.init), mean
        )

        return SepCEMState(
            mean=mean,
            variance=variance,
            cov_eps=jnp.float32(self.cov_eps_schedule.init),
            key=key,
        )

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
        state = state.replace(key=key, pop=pop)

        return pop, state

    def tell(
        self, state: SepCEMState, fitnesses: chex.Array
    ) -> tuple[PyTreeDict, SepCEMState]:
        # fitness: episode_return, higher is better
        elite_indices = jax.lax.top_k(fitnesses, self.num_elites)[1]

        mean = jtu.tree_map(
            lambda x: weight_sum(x[elite_indices], self.elite_weights),
            state.pop,
        )

        def var_update(m, x):
            x_norm = jnp.square(x[elite_indices] - m)
            # TODO: do we need extra division by num_elites mentioned in CEM-RL?
            return weight_sum(x_norm, self.elite_weights) + state.cov_eps

        variance = jtu.tree_map(
            var_update,
            state.mean,  # old mean
            state.pop,
        )

        cov_eps = optax.incremental_update(
            self.cov_eps_schedule.final, state.cov_eps, self.cov_eps_schedule.decay
        )

        return PyTreeDict(), state.replace(
            mean=mean, variance=variance, cov_eps=cov_eps, pop=None
        )
