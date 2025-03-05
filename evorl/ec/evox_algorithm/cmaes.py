import math
import numpy as np

import jax
import jax.numpy as jnp

from evox import Algorithm, State
from .sort_utils import sort_by_key


class CMAES(Algorithm):
    """CMA-ES.

    Paper: [Completely Derandomized Self-Adaptation in Evolution Strategies](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf)
    """

    def __init__(
        self,
        center_init,
        init_stdev,
        pop_size=None,
        mu=None,
        recombination_weights=None,
        cm: float = 1.0,
        alpha_cov: float = 2.0,
        delay_decomp: bool = True,
    ):
        """This implementation follows `The CMA Evolution Strategy: A Tutorial <https://arxiv.org/pdf/1604.00772.pdf>`_.

        .. note::
            CMA-ES involves eigendecomposition,
            which introduces relatively large numerical error,
            and may lead to non-deterministic behavior on different hardware backends.
        """
        self.center_init = center_init
        assert init_stdev > 0, "Expect variance to be a non-negative float"
        self.init_stdev = init_stdev
        self.dim = center_init.shape[0]
        self.cm = cm

        if pop_size is None:
            # auto
            self.pop_size = 4 + math.floor(3 * math.log(self.dim))
        else:
            self.pop_size = pop_size

        if mu is None:
            self.mu = self.pop_size // 2
        else:
            assert mu <= self.pop_size, "mu should be less than or equal to pop_size"
            self.mu = mu

        if recombination_weights is None:
            # log(pop/2+1) - log(i) for i=1,2,...,mu

            # Note: equivalent when mu = pop_size / 2
            # weights = jnp.log(self.mu + 0.5) - jnp.log(jnp.arange(1, self.mu + 1))
            weights = np.log((self.pop_size + 1) / 2) - np.log(
                np.arange(1, self.mu + 1)
            )
            self.weights = weights / np.sum(weights)
        else:
            assert recombination_weights.shape[0] == self.mu, (
                "recombination_weights' length must match mu"
            )
            # currently, we don't support negative weights
            assert (recombination_weights[1:] <= recombination_weights[:-1]).all(), (
                "recombination_weights must be non-increasing"
            )
            assert jnp.abs(jnp.sum(recombination_weights) - 1) < 1e-6, (
                "sum of recombination_weights must be 1"
            )
            assert (recombination_weights > 0).all(), (
                "recombination_weights must be positive"
            )
            self.weights = np.asarray(recombination_weights, dtype=np.float32)

        self.mueff = (
            np.sum(np.abs(self.weights)) ** 2 / np.sum(self.weights**2)
        ).tolist()

        # time constant for cumulation for C
        self.cc = (4 + self.mueff / self.dim) / (
            self.dim + 4 + 2 * self.mueff / self.dim
        )

        # learning rate for rank-one update of C
        self.alpha_cov = alpha_cov
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

        if delay_decomp:
            self.delay_decomp_iters = max(
                1, math.floor(1 / ((self.c1 + self.cmu) * self.dim * 10))
            )
        else:
            self.delay_decomp_iters = 0

    def setup(self, key):
        pc = jnp.zeros((self.dim,))
        ps = jnp.zeros((self.dim,))
        B = jnp.eye(self.dim)
        D = jnp.ones((self.dim,))
        C = jnp.eye(self.dim)
        return State(
            pc=pc,  # env path for C
            ps=ps,  # env path for sigma
            B=B,
            D=D,
            C=C,
            count_iter=jnp.zeros((), dtype=jnp.int32),
            mean=jnp.asarray(self.center_init, dtype=jnp.float32),
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
        return population, new_state

    def tell(self, state, fitness):
        fitness, population = sort_by_key(fitness, state.population)

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

        return state.replace(
            mean=mean,
            ps=ps,
            pc=pc,
            C=C,
            sigma=sigma,
        )

    def _update_ps(self, ps, B, D, y_w):
        invsqrtC = (B / D) @ B.T  # C^(-1/2)=B*D^(-1)*B^T

        return (1 - self.cs) * ps + jnp.sqrt(self.cs * (2 - self.cs) * self.mueff) * (
            invsqrtC @ y_w
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

    def setup(self, key):
        pc = jnp.zeros((self.dim,))
        ps = jnp.zeros((self.dim,))
        C = jnp.ones((self.dim,))
        return State(
            pc=pc,
            ps=ps,
            C=C,
            count_iter=0,
            mean=self.center_init,
            sigma=self.init_stdev,
            key=key,
        )

    def ask(self, state):
        key, sample_key = jax.random.split(state.key)
        noise = jax.random.normal(sample_key, (self.pop_size, self.dim))
        population = state.mean + state.sigma * jnp.sqrt(state.C) * noise
        new_state = state.replace(
            population=population, count_iter=state.count_iter + 1, key=key
        )
        return population, new_state

    def tell(self, state, fitness):
        fitness, population = sort_by_key(fitness, state.population)

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

        return state.replace(mean=mean, ps=ps, pc=pc, C=C, sigma=sigma)

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
