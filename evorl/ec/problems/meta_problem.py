import logging
from typing import List, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from evox import Problem

from evorl.types import State
from evorl.workflows import Workflow

MetaInfo = chex.ArrayTree

logger = logging.getLogger(__name__)


class AbstractMetaProblem(Problem):
    def evaluate(self, state: State, pop: chex.ArrayTree) -> tuple[MetaInfo, State]:
        """

        return:
            metadata: [pop_size, ...]
        """
        raise NotImplementedError


class MetaProblem(AbstractMetaProblem):
    """
    Single Workflow MetaProblem.
    Design choice: don't parallel MetaProblem, parallel its sub workflow.
    """

    def __init__(
        self,
        workflow,
        workflow_train_steps: int,
        workflow_eval_interval: int,
        num_meta_evals: int = 1,
        parallel_meta_evaluation: bool = False,
    ):
        """
        Args:
            workflow: Sub Workflow to optimize
            workflow_train_steps: number of steps to train the workflow for one individual
            workflow_eval_interval: after every k train steps, evaluate the workflow
            num_meta_evals: number of evaluations for one individual, i.e., number of training of the workflow
            parallel_meta_evaluation: whether to parallel the training of the workflow over the population.
                Caution: this will consume heavy (GPU) memory.
        """
        self.workflow = workflow
        self.workflow_train_steps = workflow_train_steps
        self.workflow_eval_interval = workflow_eval_interval
        self.num_meta_evals = num_meta_evals
        self.parallel_meta_evaluation = parallel_meta_evaluation

        if workflow_train_steps % workflow_eval_interval != 0:
            workflow_train_steps = (
                workflow_train_steps // workflow_eval_interval * workflow_eval_interval
            )
            logger.warning(
                f"workflow_train_steps ({self.workflow_train_steps}) should be divisible by workflow_eval_interval ({self.workflow_eval_interval}), set new workflow_train_steps to {workflow_train_steps}"
            )
            self.workflow_train_steps = workflow_train_steps

    def setup(self, key: chex.PRNGKey):
        return State(
            key=key,
            num_evaluated_workflows=0,  # number of trainings for the workflow
        )

    def evaluate(self, state: State, pop: chex.ArrayTree) -> tuple[MetaInfo, State]:
        """
        return:
            metadata: [pop_size, ...]
            state: MetaProblem State
        """
        key, init_key = jax.random.split(state.key)

        pop_size = jtu.tree_leaves(pop)[0].shape[0]

        if self.parallel_meta_evaluation:
            metrics = jax.vmap(self.evaluate_individual)(
                jax.random.split(init_key, pop_size), pop
            )
        else:
            metrics = jax.lax.map(
                lambda x: self.evaluate_individual(*x),
                (jax.random.split(init_key, pop_size), pop),
            )

        num_evaluated_workflows = (
            state.num_evaluated_workflows + pop_size * self.num_meta_evals
        )

        state = state.replace(key=key, num_evaluated_workflows=num_evaluated_workflows)

        # Note: we drop the workflow_state.
        return metrics, state

    def evaluate_individual(self, init_key, individual) -> MetaInfo:

        def _one_time_evaluate(key, _):
            key, init_key = jax.random.split(key)
            workflow_state = self.workflow.init(init_key)
            workflow_state = self._apply_indv_to_workflow_state(
                workflow_state, individual
            )

            metrics, workflow_state = self._run_workflow(
                workflow_state, self.workflow_train_steps
            )

            return key, metrics

        _, metrics = jax.lax.scan(
            _one_time_evaluate, init_key, (), length=self.num_meta_evals
        )

        def _summarize_metrics(metrics):
            return jtu.tree_map(lambda x: x.mean(axis=0), metrics)

        return _summarize_metrics(metrics)

    def _apply_indv_to_workflow_state(self, workflow_state, individual):
        """
        Merge your individual into the workflow_state.
        Eg: apply hyperparameters to the workflow_state
        """
        raise NotImplementedError

    def _run_workflow(self, workflow_state):
        num_evals = self.workflow_train_steps // self.workflow_eval_interval

        def _n_steps(wf_state, _):
            def _one_step(wf_state, _):
                train_metrics, wf_state = self.workflow.step(wf_state)
                return wf_state, train_metrics

            wf_state, train_metrics = jax.lax.scan(
                _one_step, wf_state, (), length=self.workflow_eval_interval
            )
            # report the last train_metrics
            train_metrics = jtu.tree_map(lambda x: x[-1], train_metrics)

            eval_metrics, wf_state = self.workflow.evaluate(wf_state)

            return wf_state, (train_metrics, eval_metrics)

        # note: n should be fixed
        workflow_state, (train_metrics_trajectory, eval_metric_trajectory) = (
            jax.lax.scan(_n_steps, workflow_state, (), length=num_evals)
        )

        return self._summarize_metrics(train_metrics_trajectory), workflow_state

    def _summarize_metrics(self, train_metrics_trajectory, eval_metrics_trajectory):
        """
        summarize the metrics trajectory
        """
        raise NotImplementedError

    def _get_metadata(
        self, workflow_state: State, train_metrics: chex.ArrayTree
    ) -> State:
        """
        get metadata from state
        """
        raise NotImplementedError
