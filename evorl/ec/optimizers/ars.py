import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.types import PyTreeData, PyTreeDict, Params, pytree_field
from evorl.utils.jax_utils import rng_split_like_tree

from .utils import weight_sum, optimizer_map
from .ec_optimizer import EvoOptimizer, ECState


class ARSState(PyTreeData):
    """State of the ARS."""

    mean: chex.ArrayTree
    opt_state: optax.OptState
    key: chex.PRNGKey
    noise: None | chex.ArrayTree = None


class ARS(EvoOptimizer):
    """Augmented Random Search.

    Paper: [Simple random search of static linear policies is competitive for reinforcement learning](https://proceedings.neurips.cc/paper_files/paper/2018/file/7634ea65a4e6d9041cfd3f7de18e334a-Paper.pdf)
    """

    pop_size: int
    num_elites: int
    lr: float
    noise_std: float
    fitness_std_eps: float = 1e-8
    optimizer_name: str = "sgd"

    optimizer: optax.GradientTransformation = pytree_field(static=True, init=False)

    def __post_init__(self):
        assert self.pop_size > 0 and self.pop_size % 2 == 0, (
            "pop_size must be positive even number"
        )

        self.optimizer = optimizer_map[self.optimizer_name](learning_rate=self.lr)

    def init(self, mean: Params, key: chex.PRNGKey) -> ARSState:
        opt_state = self.optimizer.init(mean)
        return ARSState(mean=mean, opt_state=opt_state, key=key)

    def ask(self, state: ARSState) -> tuple[Params, ECState]:
        key, sample_key = jax.random.split(state.key)
        sample_keys = rng_split_like_tree(sample_key, state.mean)

        half_noise = jtu.tree_map(
            lambda x, k: jax.random.normal(k, shape=(self.pop_size // 2, *x.shape)),
            state.mean,
            sample_keys,
        )
        noise = jtu.tree_map(lambda z: jnp.concatenate([z, -z], axis=0), half_noise)

        pop = jtu.tree_map(
            lambda m, z: m + self.noise_std * z,
            state.mean,
            noise,
        )
        return pop, state.replace(key=key, noise=half_noise)

    def tell(
        self, state: ARSState, fitnesses: chex.Array
    ) -> tuple[PyTreeDict, ARSState]:
        half_pop_size = self.pop_size // 2

        fit_p = fitnesses[:half_pop_size]  # r_positive
        fit_n = fitnesses[half_pop_size:]  # r_negtive
        elite_indices = jax.lax.top_k(jnp.maximum(fit_p, fit_n), self.num_elites)[1]

        fitnesses_elite = jnp.concatenate([fit_p[elite_indices], fit_n[elite_indices]])
        # Add small constant to ensure non-zero division stability
        fitness_std = jnp.std(fitnesses_elite) + self.fitness_std_eps

        fit_diff = (fit_p[elite_indices] - fit_n[elite_indices]) / fitness_std

        grad = jtu.tree_map(
            # Note: we need additional "-1.0" since we are maximizing the fitness
            lambda z: (-weight_sum(z[elite_indices], fit_diff) / (self.num_elites)),
            state.noise,
        )

        update, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, update)

        ec_metrics = PyTreeDict(fitness_std=fitness_std)

        return ec_metrics, state.replace(mean=mean, opt_state=opt_state, noise=None)
