"""Single-objective CSO example using the shared EvoX workflow."""

import torch
from evox.algorithms import CSO
from omegaconf import DictConfig
from collections.abc import Callable

from .evox_algorithm_wrapper import EvoXAlgorithmWrapper
from .evox_workflow import EvoXWorkflow


class CSOWorkflow(EvoXWorkflow):
    """Optimize scalar policy returns with EvoX's Competitive Swarm Optimizer."""

    @classmethod
    def name(cls) -> str:
        return "CSO-EvoX"

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
        """Construct CSO with explicit policy bounds and its `phi` parameter."""
        if num_objectives is not None:
            raise ValueError("CSO requires scalar rewards")
        lb = torch.full((param_dim,), config.agent_network.lb, device=device)
        ub = torch.full((param_dim,), config.agent_network.ub, device=device)
        return EvoXAlgorithmWrapper(
            algorithm_cls=CSO,
            algorithm_kwargs=dict(
                pop_size=config.pop_size,
                lb=lb,
                ub=ub,
                phi=config.phi,
                device=device,
            ),
            device=device,
            compile=config.torch_compile,
            evaluate=evaluate,
            static_tensors=dict(lb=lb[None, :], ub=ub[None, :]),
        )
