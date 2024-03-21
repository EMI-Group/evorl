import warnings
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp

from evorl.utils.jax_utils import jit_method

from evox import Algorithm, Problem, State, Monitor
from evox.utils import parse_opt_direction, algorithm_has_init_ask
from evox import Stateful
from .workflow import Workflow


class EAWorkflow(Workflow):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Union[Problem, List[Problem]],
        monitors: List[Monitor] = [],
        opt_direction: Union[str, List[str]] = "min",
        sol_transforms: List[Callable] = [],
        fit_transforms: List[Callable] = [],
        jit_problem: bool = True,
        num_objectives: Optional[int] = None,
        monitor=None,
    ):
        """
        Parameters
        ----------
        algorithm
            The algorithm.
        problem
            The problem.
        monitor
            Optional monitor(s).
            Configure a single monitor or a list of monitors.
            The monitors will be called in the order of the list.
        opt_direction
            The optimization direction, can be either "min" or "max"
            or a list of "min"/"max" to specific the direction for each objective.
        sol_transform
            Optional candidate solution transform function,
            usually used to decode the candidate solution
            into the format that can be understood by the problem.
            Should be a list of functions,
            and the functions will be applied in the order of the list.
        fit_transforms
            Optional fitness transform function.
            usually used to apply fitness shaping.
            Should be a list of functions,
            and the functions will be applied in the order of the list.
        jit_problem
            If the problem can be jit compiled by JAX or not.
            Default to True.
        num_objectives
            Number of objectives.
            When the problem can be jit compiled, this field is not needed.
            When the problem cannot be jit compiled, this field should be set,
            if not, default to 1.
        """
        self.algorithm = algorithm
        self.problem = problem
        self.monitors = monitors
        if monitor is not None:
            warnings.warn(
                "`monitor` is deprecated, use the `monitors` parameter with a list of monitors instead",
                DeprecationWarning,
            )
            self.monitors = [monitor]
        self.registered_hooks = {
            "pre_step": [],
            "pre_ask": [],
            "post_ask": [],
            "pre_eval": [],
            "post_eval": [],
            "pre_tell": [],
            "post_tell": [],
            "post_step": [],
        }
        for monitor in self.monitors:
            hooks = monitor.hooks()
            for hook in hooks:
                self.registered_hooks[hook].append(monitor)

        self.opt_direction = parse_opt_direction(opt_direction)
        for monitor in self.monitors:
            monitor.set_opt_direction(self.opt_direction)

        self.sol_transforms = sol_transforms

        self.fit_transforms = fit_transforms
        self.jit_problem = jit_problem
        self.num_objectives = num_objectives
        self.distributed_step = False
        if jit_problem is False and self.num_objectives is None:
            warnings.warn(
                (
                    "Using external problem "
                    "but num_objectives isn't set "
                    "assuming to be 1."
                )
            )
            self.num_objectives = 1

    def candidate_generation(self, state, is_init):
        if is_init:
            ask = self.algorithm.init_ask
        else:
            ask = self.algorithm.ask

        cand_sol, state = ask(state)

        return cand_sol, state

    def candidate_evaluation(self, state, transformed_cand_sol):
        if self.jit_problem:
            fitness, state = self.problem.evaluate(state, transformed_cand_sol)
        else:
            pass
            # TODO: get cand_sol_size from state
            # if self.num_objectives == 1:
            #     fit_shape = (cand_sol_size,)
            # else:
            #     fit_shape = (cand_sol_size, self.num_objectives)
            # fitness, state = pure_callback(
            #     self.problem.evaluate,
            #     (
            #         jax.ShapeDtypeStruct(fit_shape, dtype=jnp.float32),
            #         state,
            #     ),
            #     state,
            #     transformed_cand_sol,
            # )

        return fitness, state

    def learn_one_step(self, state, transformed_fitness, is_init):
        if is_init:
            tell = self.algorithm.init_tell
        else:
            tell = self.algorithm.tell

        state = tell(state, transformed_fitness)
        return state

    # a prototype step function
    # will be then wrapped to get _step
    # We are doing this as a workaround for JAX's static shape requirement
    # Since init_ask and ask can return different shape
    # and jax.lax.cond requires the same shape from two different branches
    # we can only apply lax.cond outside of each `step`
    def _proto_step(self, is_init, state):
        # ======== candidate generations ========
        for monitor in self.registered_hooks["pre_ask"]:
            monitor.pre_ask(state)

        # candidate solution
        cand_sol, state = self.candidate_generation(state, is_init)

        for monitor in self.registered_hooks["post_ask"]:
            monitor.post_ask(state, cand_sol)

        transformed_cand_sol = cand_sol
        for transform in self.sol_transforms:
            transformed_cand_sol = transform(transformed_cand_sol)

        # =========== problem evaluation ===========
        for monitor in self.registered_hooks["pre_eval"]:
            monitor.pre_eval(state, cand_sol, transformed_cand_sol)

        fitness, state = self.candidate_evaluation(state, transformed_cand_sol)
        fitness = fitness * self.opt_direction

        for monitor in self.registered_hooks["post_eval"]:
            monitor.post_eval(state, cand_sol, transformed_cand_sol, fitness)

        # =========== algorithm iteration ===========

        transformed_fitness = fitness
        for transform in self.fit_transforms:
            transformed_fitness = transform(transformed_fitness)

        for monitor in self.registered_hooks["pre_tell"]:
            monitor.pre_tell(
                state, cand_sol, transformed_cand_sol, fitness, transformed_fitness
            )

        state = self.learn_one_step(state, transformed_fitness, is_init)

        for monitor in self.registered_hooks["post_tell"]:
            monitor.post_tell(state)

        return state.update(generation=state.generation + 1)

    # wrap around _proto_step
    # to handle init_ask and init_tell
    @jit_method
    def _step(self, state):
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

    def setup(self, key):
        return State(generation=0)

    def step(self, state):
        for monitor in self.registered_hooks["pre_step"]:
            monitor.pre_step(state)

        state = self._step(state)

        for monitor in self.registered_hooks["post_step"]:
            monitor.post_step(state)

        return state
