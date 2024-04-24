import warnings
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import chex

from evorl.utils.jax_utils import jit_method

from evox import Algorithm, Problem, State, Monitor
from evox.utils import parse_opt_direction, algorithm_has_init_ask
from evox import use_state
# from .workflow import Workflow
from evox import Workflow


class ECWorkflow(Workflow):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        monitors: List[Monitor] = [],
        opt_direction: Union[str, List[str]] = "min",
        sol_transforms: List[Callable] = [],
        fit_transforms: List[Callable] = [],
    ):
        self.algorithm = algorithm
        self.problem = problem
        self.monitors = monitors

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

    # a prototype step function
    # will be then wrapped to get _step
    # We are doing this as a workaround for JAX's static shape requirement
    # Since init_ask and ask can return different shape
    # and jax.lax.cond requires the same shape from two different branches
    # we can only apply lax.cond outside of each `step`
    def _proto_step(self, is_init: bool, state: State) -> State:
        # ======== candidate generations ========
        for monitor in self.registered_hooks["pre_ask"]:
            monitor.pre_ask(state)

        if is_init:
            ask = self.algorithm.init_ask
        else:
            ask = self.algorithm.ask

        cands, state = use_state(ask)(state)

        for monitor in self.registered_hooks["post_ask"]:
            monitor.post_ask(state, cands)

        transformed_cands = cands
        for transform in self.sol_transforms:
            transformed_cands = transform(transformed_cands)

        # =========== problem evaluation ===========
        for monitor in self.registered_hooks["pre_eval"]:
            monitor.pre_eval(state, cands, transformed_cands)

        fitness, state = use_state(self.problem.evaluate)(
            state, transformed_cands)
        fitness = fitness * self.opt_direction

        for monitor in self.registered_hooks["post_eval"]:
            monitor.post_eval(state, cands, transformed_cands, fitness)

        # =========== algorithm iteration ===========

        transformed_fitness = fitness
        for transform in self.fit_transforms:
            transformed_fitness = transform(transformed_fitness)

        for monitor in self.registered_hooks["pre_tell"]:
            monitor.pre_tell(
                state, cands, transformed_cands, fitness, transformed_fitness
            )

        if is_init:
            tell = self.algorithm.init_tell
        else:
            tell = self.algorithm.tell

        state = use_state(tell)(state, transformed_fitness)

        for monitor in self.registered_hooks["post_tell"]:
            monitor.post_tell(state)

        return state.replace(generation=state.generation + 1)

    # wrap around _proto_step
    # to handle init_ask and init_tell
    @jit_method(static_argnums=(0,))
    def _step(self, state: State) -> State:
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

    def setup(self, key: chex.PRNGKey) -> State:
        return State(generation=0)

    def step(self, state: State) -> State:
        for monitor in self.registered_hooks["pre_step"]:
            monitor.pre_step(state)

        state = self._step(state)

        for monitor in self.registered_hooks["post_step"]:
            monitor.post_step(state)

        return state
