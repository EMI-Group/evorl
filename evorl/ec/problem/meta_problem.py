import jax

from evox import Problem, State
import chex
from typing import Tuple, List
from evorl.workflows import Workflow


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

        return self._get_metadata(workflow_state)

    def _get_metadata(self, workflow_state: State) -> State:
        """
            get metadata from state
        """
        raise NotImplementedError


class PBTProblem(AbstractMetaProblem):
    def __init__(self, workflows: List[Workflow]):
        self.workflows = workflows

    def setup(self, key: chex.PRNGKey):
        num_workflows = len(self.workflows)
        key, wf_init_key = jax.random.split(key)
        workflow_states = [
            wf.init(_key)
            for wf, _key in zip(
                self.workflows,
                jax.random.split(wf_init_key, num_workflows))
        ]

        return State(
            key=key,
            workflow_states=workflow_states,
            interations=0,  # number of iterations in the workflow
            n_steps_per_iter=[1]*num_workflows
        )

    def _run_wf(self, workflow: Workflow, workflow_state, n: int):
        def _n_steps(wf_state, _):
            train_metrics, wf_state = workflow.step(wf_state)
            return wf_state, None

        workflow_state, _ = jax.lax.scan(
            _n_steps, workflow_state, (), length=n)

        return workflow_state
