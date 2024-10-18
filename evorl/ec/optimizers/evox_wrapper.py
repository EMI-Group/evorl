from typing import Protocol


import chex

from evox import (
    State as EvoXState,
    Algorithm,
    has_init_ask,
    has_init_tell,
)

from evorl.types import PyTreeData, pytree_field, Params

from .ec_optimizer import EvoOptimizer, ECState


class TransformFn(Protocol):
    """
    Convert EvoX's flat individual to Flax Params
    """

    def __call__(self, flat_x: chex.Array) -> Params: ...


class EvoXAlgoState(PyTreeData):
    algo_state: EvoXState
    first_step: bool = pytree_field(pytree_node=False)


class EvoXAlgorithmAdapter(EvoOptimizer):
    """
    Adapter class to convert EvoX algorithms to EvoRL optimizers.
    """

    algorithm: Algorithm
    transform_fn: TransformFn

    def __init__(
        self,
        algorithm: Algorithm,
    ):
        self.algorithm = algorithm

    def init(self, key: chex.PRNGKey) -> EvoXAlgoState:
        algo_state = self.algorithm.init(key)

        return EvoXAlgoState(algo_state=algo_state, first_step=True)

    def tell(
        self, state: EvoXAlgoState, xs: chex.ArrayTree, fitnesses: chex.Array
    ) -> EvoXAlgoState:
        if has_init_tell(self.algorithm) and state.first_step:
            tell = self.algorithm.init_tell
        else:
            tell = self.algorithm.tell

        algo_state = tell(state.algo_state, fitnesses)

        return state.replace(algo_state=algo_state, first_step=False)

    def ask(self, state: EvoXAlgoState) -> tuple[chex.ArrayTree, ECState]:
        if has_init_ask(self.algorithm) and state.first_step:
            ask = self.algorithm.init_ask
        else:
            ask = self.algorithm.ask

        flat_pop, algo_state = ask(state.algo_state)

        pop = self.transform_fn(flat_pop)

        return pop, state.replace(algo_state=algo_state)
