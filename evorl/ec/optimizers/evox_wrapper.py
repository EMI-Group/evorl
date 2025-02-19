import chex
from evox import (
    State as EvoXState,
    Algorithm,
    has_init_ask,
    has_init_tell,
)

from evorl.types import PyTreeData, pytree_field, PyTreeDict
from evorl.utils.ec_utils import ParamVectorSpec

from .ec_optimizer import EvoOptimizer


class EvoXAlgoState(PyTreeData):
    algo_state: EvoXState
    init_step: bool = pytree_field(static=True)


class EvoXAlgorithmAdapter(EvoOptimizer):
    """Adapter class to convert EvoX algorithms to EvoRL optimizers."""

    algorithm: Algorithm
    param_vec_spec: ParamVectorSpec

    def init(self, key: chex.PRNGKey) -> EvoXAlgoState:
        algo_state = self.algorithm.init(key)

        if has_init_tell(self.algorithm):
            assert has_init_ask(self.algorithm)
            init_step = True
        else:
            init_step = False

        return EvoXAlgoState(algo_state=algo_state, init_step=init_step)

    def ask(self, state: EvoXAlgoState) -> tuple[chex.ArrayTree, EvoXAlgoState]:
        if has_init_ask(self.algorithm) and state.init_step:
            ask = self.algorithm.init_ask
        else:
            ask = self.algorithm.ask

        flat_pop, algo_state = ask(state.algo_state)

        pop = self.param_vec_spec.to_tree(flat_pop)

        return pop, state.replace(algo_state=algo_state)

    def tell(
        self, state: EvoXAlgoState, fitnesses: chex.Array
    ) -> tuple[PyTreeDict, EvoXAlgoState]:
        if has_init_tell(self.algorithm) and state.init_step:
            tell = self.algorithm.init_tell
        else:
            tell = self.algorithm.tell

        # Note: Evox's Algorithms minimize the fitness
        algo_state = tell(state.algo_state, -fitnesses)

        return PyTreeDict(), state.replace(algo_state=algo_state, init_step=False)
