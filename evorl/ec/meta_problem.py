import jax

from evox import Problem, State
import chex
from typing import Tuple


class AbstractMetaProblem(Problem):
    def setup(self, key: chex.PRNGKey):
        workflow_state = self.workflow.init(key)

        return State(
            key=key,
            workflow_state=workflow_state
        )

    def evaluate(self, state: State, pop: chex.ArrayTree) -> Tuple[chex.ArrayTree, State]:
        """

            return:
                metadata: [pop_size, ...]
        """
        raise NotImplementedError

    def _get_metadata(self, state: State) -> State:
        """
            get metadata from state
        """
        raise NotImplementedError


class MetaProblem(AbstractMetaProblem):
    """

    """
    def __init__(self, workflow):
        self.workflow = workflow

    def setup(self, key: chex.PRNGKey):
       



        return State(
            key=key,
            interations=0,  # number of iterations in the workflow

            # workflow_state=workflow_state
        )

    def evaluate(self, state: State, pop: chex.ArrayTree) -> Tuple[chex.ArrayTree, State]:
        """

            return:
                metadata: [pop_size, ...]
        """
        key, init_key = jax.random.split(state.key)
        workflow_state = self.workflow.init(init_key)
        workflow.learn()



    def _get_metadata(self, state: State) -> State:
        """
            get metadata from state
        """
        raise NotImplementedError


class PBTProblem(AbstractMetaProblem):
    pass


