"""Multi-objective RVEA example using the shared EvoX workflow."""

import torch
from evox.algorithms import RVEA as EvoXRVEA
from evox.core import Mutable
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp
from omegaconf import DictConfig
from collections.abc import Callable

from .evox_algorithm_wrapper import EvoXAlgorithmWrapper
from .evox_workflow import EvoXWorkflow


class RVEA(EvoXRVEA):
    """EvoX RVEA with int32 generation counters for JAX interoperability.

    EvoX 1.4.0 uses int64 for both counters and for the increment in `step`.
    Keep the native operators and generation schedule, using int32 throughout.
    """

    def __init__(self, *, max_gen: int = 100, **kwargs):
        super().__init__(
            max_gen=torch.tensor(
                max_gen, dtype=torch.int32, device=kwargs.get("device")
            ),
            **kwargs,
        )
        self.gen = Mutable(torch.zeros((), dtype=torch.int32, device=self.pop.device))

    def step(self):
        # A Python integer preserves int32; native torch.tensor(1) promotes it.
        self.gen = self.gen + 1
        pop = self._mating_pool()
        crossovered = self.crossover(pop)
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        survivor, survivor_fit = self.selection(
            torch.cat([self.pop, offspring], dim=0),
            torch.cat([self.fit, off_fit], dim=0),
            self.reference_vector,
            (self.gen / self.max_gen) ** self.alpha,
        )
        self._update_pop_and_rv(survivor, survivor_fit)


class RVEAWorkflow(EvoXWorkflow):
    """Optimize vector policy returns with EvoX's reference-vector algorithm."""

    @classmethod
    def name(cls) -> str:
        return "RVEA-EvoX"

    @classmethod
    def _build_algorithm(
        cls,
        config: DictConfig,
        *,
        param_dim: int,
        num_objectives: int | None,
        device: torch.device,
        evaluate: Callable,
    ) -> EvoXAlgorithmWrapper:
        """Construct RVEA for at least two reward objectives.

        Setup evaluates the initial population. The `num_iters` regular
        updates define RVEA's penalty schedule. `max_gen` stays positive when
        no training updates are requested.
        """
        if num_objectives is None or num_objectives < 2:
            raise ValueError(
                "RVEA requires vector rewards with at least two objectives"
            )
        lb = torch.full((param_dim,), config.agent_network.lb, device=device)
        ub = torch.full((param_dim,), config.agent_network.ub, device=device)
        reference_vectors, _ = uniform_sampling(config.pop_size, num_objectives)
        return EvoXAlgorithmWrapper(
            algorithm_cls=RVEA,
            algorithm_kwargs=dict(
                pop_size=config.pop_size,
                n_objs=num_objectives,
                lb=lb,
                ub=ub,
                alpha=config.alpha,
                fr=config.fr,
                max_gen=max(1, config.num_iters),
                device=device,
            ),
            device=device,
            compile=config.torch_compile,
            evaluate=evaluate,
            static_tensors=dict(lb=lb, ub=ub, init_v=reference_vectors.to(device)),
        )
