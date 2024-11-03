import copy
from optax.schedules import InjectStatefulHyperparamsState


def deepcopy_opt_state(state: InjectStatefulHyperparamsState):
    assert isinstance(state, InjectStatefulHyperparamsState)

    return InjectStatefulHyperparamsState(
        count=state.count,
        hyperparams=copy.deepcopy(state.hyperparams),
        hyperparams_states=state.hyperparams_states,
        inner_state=state.inner_state,
    )
