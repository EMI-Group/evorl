import chex

from evox import (
    State as EvoXState,
    Algorithm,
    has_init_ask,
    has_init_tell,
)

from evorl.types import PyTreeData, pytree_field

from .ec_optimizer import EvoOptimizer


class EvoXAlgoState(PyTreeData):
    algo_state: EvoXState
    first_step: bool = pytree_field(pytree_node=False)


class EvoXAlgorithmAdapter(EvoOptimizer):
    """
    Adapter class to convert EvoX algorithms to EvoRL optimizers.
    """

    def __init__(self, algorithm: Algorithm):
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

    def ask(self, state: EvoXAlgoState, key: chex.PRNGKey) -> chex.ArrayTree:
        if has_init_ask(self.algorithm) and state.first_step:
            ask = self.algorithm.init_ask
        else:
            ask = self.algorithm.ask

        # algo_state = state.algo_state
        # try:
        #     algo_state = algo_state.replace(key=key)
        # except Exception:
        #     pass
        algo_state = ask(state.algo_state)

        return state.replace(algo_state=algo_state)
