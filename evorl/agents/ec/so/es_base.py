import jax
import jax.numpy as jnp
import chex
from typing import Union
from collections.abc import Callable, Sequence
from omegaconf import DictConfig, OmegaConf

from evox import Algorithm, Problem

from evorl.agents import Agent
from evorl.evaluator import Evaluator
from evorl.workflows import ECWorkflow
from evorl.metrics import EvaluateMetric
from evorl.evaluator import Evaluator
from evorl.types import State


class ESBaseWorkflow(ECWorkflow):
    def __init__(
        self,
        config: DictConfig,
        agent: Agent,
        evaluator: Evaluator,
        algorithm: Algorithm,
        problem: Problem,
        opt_direction: Union[str, Sequence[str]] = 'max',
        candidate_transforms: Sequence[Callable] = (),
        fitness_transforms: Sequence[Callable] = (),
    ):
        super(ESBaseWorkflow, self).__init__(
            config=config, 
            agent=agent,
            algorithm=algorithm,
            problem=problem,
            opt_direction=opt_direction,
            candidate_transforms=candidate_transforms,
            fitness_transforms=fitness_transforms
        )

        # An extra evalutor for pop_center
        self.evaluator = evaluator


    def evaluate(self, state: State) -> tuple[EvaluateMetric, State]:
        raise NotImplementedError
    
    @classmethod
    def enable_jit(cls) -> None:
        super().enable_jit()
        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))

    @classmethod
    def enable_pmap(cls, axis_name) -> None:
        super().enable_pmap(axis_name)
        cls.evaluate = jax.pmap(
            cls.evaluate, axis_name, static_broadcasted_argnums=(0,)
        )