import math
import numpy as np

import jax
import jax.numpy as jnp

from evorl.types import Params, PyTreeData, PyTreeDict, pytree_field
from evorl.utils.ec_utils import ParamVectorSpec
from .ec_optimizer import EvoOptimizer


class CMAESState(PyTreeData):
    """Mean, covariance, evolution paths and pending candidates."""

    mean: jax.Array
    sigma: jax.Array
    pc: jax.Array
    ps: jax.Array
    C: jax.Array
    B: jax.Array | None
    D: jax.Array | None
    count_iter: jax.Array
    population: jax.Array
    key: jax.Array


class CMAES(EvoOptimizer):
    """CMA-ES maximizing scalar fitness, with a full covariance matrix.

    `init(mean, key)` accepts policy parameters as a PyTree. The state stores
    flattened parameters for covariance updates; `ask` returns parameter trees
    with a leading population dimension, using `param_vec_spec` to restore them.
    This is the positive-weight variant in Appendix C of the tutorial. Use
    `num_elites <= pop_size // 2` with the population-based logarithmic weights.

    Reference: https://arxiv.org/abs/1604.00772.
    """

    pop_size: int
    num_elites: int
    init_stdev: float
    param_vec_spec: ParamVectorSpec = pytree_field(static=True)
    cm: float = 1.0
    alpha_cov: float = 2.0
    delay_decomp: bool = True

    def __post_init__(self):
        if self.init_stdev <= 0:
            raise ValueError("init_stdev must be positive")
        if not 1 <= self.num_elites <= self.pop_size // 2:
            raise ValueError("num_elites must be between 1 and pop_size // 2")
        self.dim = self.param_vec_spec.vec_size
        self.mu = self.num_elites
        weights = np.log((self.pop_size + 1) / 2) - np.log(np.arange(1, self.mu + 1))
        weights /= weights.sum()
        self.mueff = float(1 / np.sum(weights**2))
        self.weights = jnp.asarray(weights, dtype=jnp.float32)

        # time constant for cumulation for C
        self.cc = (4 + self.mueff / self.dim) / (
            self.dim + 4 + 2 * self.mueff / self.dim
        )

        # learning rate for rank-one update of C
        alpha_cov = self.alpha_cov
        self.c1 = alpha_cov / ((self.dim + 1.3) ** 2 + self.mueff)

        # learning rate for rank-μ update of C
        # Note: convert self.dim to float first to prevent overflow
        self.cmu = min(
            1 - self.c1,
            (
                alpha_cov
                * (self.mueff - 2 + 1 / self.mueff)
                / ((self.dim + 2.0) ** 2 + alpha_cov * self.mueff / 2)
            ),
        )

        # t-const for cumulation for sigma control
        self.cs = (2 + self.mueff) / (self.dim + self.mueff + 5)
        # damping for sigma
        self.damps = (
            1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        )

        self.chiN = jnp.float32(
            math.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        )

        if self.delay_decomp:
            self.delay_decomp_iters = max(
                1, math.floor(1 / ((self.c1 + self.cmu) * self.dim * 10))
            )
        else:
            self.delay_decomp_iters = 0

    def init(self, mean: Params, key: jax.Array) -> CMAESState:
        pc = jnp.zeros((self.dim,))
        ps = jnp.zeros((self.dim,))
        B = jnp.eye(self.dim)
        D = jnp.ones((self.dim,))
        C = jnp.eye(self.dim)
        return CMAESState(
            pc=pc,  # Evolution path for covariance.
            ps=ps,  # Evolution path for step size.
            B=B,
            D=D,
            C=C,
            count_iter=jnp.zeros((), dtype=jnp.int32),
            mean=jnp.asarray(self.param_vec_spec.to_vector(mean), dtype=jnp.float32),
            sigma=jnp.asarray(self.init_stdev, dtype=jnp.float32),
            key=key,
            population=jnp.empty((self.pop_size, self.dim)),
        )

    def ask(self, state):
        if self.delay_decomp_iters > 0:
            B, D, C = jax.lax.cond(
                state.count_iter % self.delay_decomp_iters == 0,
                _decompose_C,
                lambda _C: (state.B, state.D, state.C),
                state.C,
            )
        else:
            B, D, C = _decompose_C(state.C)

        key, sample_key = jax.random.split(state.key)
        noise = jax.random.normal(sample_key, (self.pop_size, self.dim))
        population = state.mean + state.sigma * noise @ (B * D).T

        new_state = state.replace(
            B=B,
            D=D,
            C=C,
            population=population,
            count_iter=state.count_iter + 1,
            key=key,
        )
        return self.param_vec_spec.to_tree(population), new_state

    def tell(self, state, fitness):
        population = state.population[jnp.argsort(fitness, descending=True)]

        updates = population[: self.mu] - state.mean
        update = self.weights @ updates
        mean = state.mean + self.cm * update
        y = updates / state.sigma  # [pop_size, dim]
        y_w = update / state.sigma  # [dim]

        ps = self._update_ps(state.ps, state.B, state.D, y_w)

        hsig = (
            jnp.linalg.norm(ps) / jnp.sqrt(1 - (1 - self.cs) ** (2 * state.count_iter))
            < (1.4 + 2 / (self.dim + 1)) * self.chiN
        ).astype(jnp.float32)

        pc = self._update_pc(state.pc, hsig, y_w)
        C = self._update_C(state.C, pc, hsig, y)
        sigma = self._update_sigma(state.sigma, ps)

        return PyTreeDict(), state.replace(
            mean=mean,
            ps=ps,
            pc=pc,
            C=C,
            sigma=sigma,
        )

    def _update_ps(self, ps, B, D, y_w):
        # Apply C^(-1/2) using matrix-vector products without constructing it.
        whitened = B @ ((B.T @ y_w) / D)
        return (1 - self.cs) * ps + jnp.sqrt(self.cs * (2 - self.cs) * self.mueff) * (
            whitened
        )

    def _update_pc(self, pc, hsig, y_w):
        return (1 - self.cc) * pc + hsig * jnp.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * y_w

    def _update_C(self, C, pc, hsig, y):
        delta_hsig = (1 - hsig) * self.cc * (2 - self.cc)
        rank_mu_update = (y.T * self.weights) @ y
        rank_one_update = jnp.outer(pc, pc)
        return (
            (1 + self.c1 * delta_hsig - self.c1 - self.cmu) * C
            + self.c1 * rank_one_update
            + self.cmu * rank_mu_update
        )

    def _update_sigma(self, sigma, ps):
        return sigma * jnp.exp(
            (self.cs / self.damps) * (jnp.linalg.norm(ps) / self.chiN - 1)
        )


def _decompose_C(C):
    C = 0.5 * (C + C.T)
    D2, B = jnp.linalg.eigh(C)
    D = jnp.sqrt(D2)
    return B, D, C


class SepCMAES(CMAES):
    """SepCMAES.

    Paper: [A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity](https://inria.hal.science/inria-00287367/document)
    """

    def init(self, mean: Params, key: jax.Array) -> CMAESState:
        pc = jnp.zeros((self.dim,))
        ps = jnp.zeros((self.dim,))
        C = jnp.ones((self.dim,))
        return CMAESState(
            pc=pc,
            ps=ps,
            C=C,
            B=None,
            D=None,
            population=jnp.zeros((self.pop_size, self.dim)),
            count_iter=jnp.zeros((), dtype=jnp.int32),
            mean=self.param_vec_spec.to_vector(mean),
            sigma=jnp.asarray(self.init_stdev, dtype=jnp.float32),
            key=key,
        )

    def ask(self, state):
        key, sample_key = jax.random.split(state.key)
        noise = jax.random.normal(sample_key, (self.pop_size, self.dim))
        population = state.mean + state.sigma * jnp.sqrt(state.C) * noise
        new_state = state.replace(
            population=population, count_iter=state.count_iter + 1, key=key
        )
        return self.param_vec_spec.to_tree(population), new_state

    def tell(self, state, fitness):
        population = state.population[jnp.argsort(fitness, descending=True)]

        updates = population[: self.mu] - state.mean
        update = self.weights @ updates
        mean = state.mean + self.cm * update
        y = updates / state.sigma  # [pop_size, dim]
        y_w = update / state.sigma  # [dim]

        ps = self._update_ps(state.ps, state.C, y_w)

        hsig = (
            jnp.linalg.norm(ps) / jnp.sqrt(1 - (1 - self.cs) ** (2 * state.count_iter))
            < (1.4 + 2 / (self.dim + 1)) * self.chiN
        ).astype(jnp.float32)

        pc = self._update_pc(state.pc, hsig, y_w)
        C = self._update_C(state.C, pc, hsig, y)
        sigma = self._update_sigma(state.sigma, ps)

        return PyTreeDict(), state.replace(mean=mean, ps=ps, pc=pc, C=C, sigma=sigma)

    def _update_ps(self, ps, C, y_w):
        return (1 - self.cs) * ps + jnp.sqrt(self.cs * (2 - self.cs) * self.mueff) * (
            y_w / jnp.sqrt(C)
        )

    def _update_C(self, C, pc, hsig, y):
        delta_hsig = (1 - hsig) * self.cc * (2 - self.cc)
        rank_mu_update = self.weights @ (y**2)
        rank_one_update = pc**2

        return (
            (1 + self.c1 * delta_hsig - self.c1 - self.cmu) * C
            + self.c1 * rank_one_update
            + self.cmu * rank_mu_update
        )
