import warnings
from functools import partial
from typing import Callable, Dict, List, Optional, Union, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import chex

from evorl.metrics import MetricBase
from evorl.utils.jax_utils import jit_method

from evox import Algorithm, Problem, State, Monitor
from evox.utils import parse_opt_direction, algorithm_has_init_ask
from evox import use_state
# from .workflow import Workflow
from evox import Workflow


class TrainMetric(MetricBase):
    objective: chex.Array

class ECWorkflow(Workflow):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        opt_direction: Union[str, List[str]] = "min",
        sol_transforms: List[Callable] = [],
        fit_transforms: List[Callable] = [],
    ):
        self.algorithm = algorithm
        self.problem = problem

        self.opt_direction = parse_opt_direction(opt_direction)

        self.sol_transforms = sol_transforms
        self.fit_transforms = fit_transforms

    # a prototype step function
    # will be then wrapped to get _step
    # We are doing this as a workaround for JAX's static shape requirement
    # Since init_ask and ask can return different shape
    # and jax.lax.cond requires the same shape from two different branches
    # we can only apply lax.cond outside of each `step`
    def _proto_step(self, is_init: bool, state: State) -> Tuple[TrainMetric,State]:
        # ======== candidate generations ========
        if is_init:
            ask = self.algorithm.init_ask
        else:
            ask = self.algorithm.ask

        cands, state = use_state(ask)(state)

        transformed_cands = cands
        for transform in self.sol_transforms:
            transformed_cands = transform(transformed_cands)

        # =========== problem evaluation ===========

        objective, state = use_state(self.problem.evaluate)(
            state, transformed_cands)
        fitness = objective * self.opt_direction

        # =========== algorithm iteration ===========

        transformed_fitness = fitness
        for transform in self.fit_transforms:
            transformed_fitness = transform(transformed_fitness)


        if is_init:
            tell = self.algorithm.init_tell
        else:
            tell = self.algorithm.tell

        state = use_state(tell)(state, transformed_fitness)

        train_metrics = TrainMetric(objective=objective)

        return train_metrics, state.replace(generation=state.generation + 1)



    def setup(self, key: chex.PRNGKey) -> State:
        return State(generation=0)

    # wrap around _proto_step
    # to handle init_ask and init_tell
    @jit_method(static_argnums=(0,))
    def step(self, state: State) -> Tuple[TrainMetric,State]:
        # probe if self.algorithm has override the init_ask function
        if algorithm_has_init_ask(self.algorithm, state):
            return jax.lax.cond(
                state.generation == 0,
                partial(self._proto_step, True),
                partial(self._proto_step, False),
                state,
            )
        else:
            return self._proto_step(False, state)
