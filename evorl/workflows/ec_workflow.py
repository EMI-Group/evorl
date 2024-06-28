from functools import partial
from typing import Callable, Dict, List, Optional, Union, Tuple, Sequence
from omegaconf import DictConfig, OmegaConf

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import chex
import orbax.checkpoint as ocp

from evox import Algorithm, Problem
from evox.workflows import StdWorkflow as EvoXWorkflow

from evorl.types import State
from evorl.metrics import MetricBase, WorkflowMetric
from evorl.utils.cfg_utils import get_output_dir
from .workflow import Workflow


class WorkflowMetric(MetricBase):
    # note: we use float32, as overflow may easily happens with int32 in EC
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_timesteps: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class TrainMetric(MetricBase):
    objectives: chex.Array


class ECWorkflow(Workflow):
    def __init__(
        self,
        config: DictConfig,
        algorithm: Algorithm,
        problem: Union[Problem, List[Problem]],
        opt_direction: Union[str, List[str]] = 'max',
        candidate_transforms: List[Callable] = [],
        fitness_transforms: List[Callable] = [],
    ):
        super(ECWorkflow, self).__init__(config)
        self._workflow = EvoXWorkflow(
            algorithm=algorithm,
            problem=problem,
            monitors=[],
            opt_direction=opt_direction,
            candidate_transforms=candidate_transforms,
            fitness_transforms=fitness_transforms,
            jit_problem=True,
        )
        self.pmap_axis_name = None

    @property
    def enable_multi_devices(self) -> bool:
        return self._workflow.pmap_axis_name is not None

    def setup_multiple_device(self, state: State, devices: Optional[Sequence[jax.Device]] = None) -> State:
        evox_state = self._workflow.enable_multi_devices(
            state.evox_state, devices)
        self.pmap_axis_name = self._workflow.pmap_axis_name
        return state.replace(evox_state=evox_state)

    def setup(self, key: chex.PRNGKey) -> State:
        evox_state = self._workflow.init(key)
        workflow_metrics = self._setup_workflow_metrics()
        return State(
            key=key,
            evox_state=evox_state,
            metrics=workflow_metrics
        )

    def _setup_workflow_metrics(self) -> MetricBase:
        """
            Customize the workflow metrics.
        """
        return WorkflowMetric()

    def step(self, state: State) -> Tuple[TrainMetric, State]:
        train_info, evox_state = self._workflow.step(state.evox_state)

        # turn back to the original objectives
        train_metrics = TrainMetric(
            objectives=train_info['fitness'] * self._workflow.opt_direction
        )

        problem_state = state.evox_state.get_child_state('problem')

        workflow_metrics = WorkflowMetric(
            sampled_episodes=state.metrics.sampled_episodes+problem_state.sampled_episodes,
            sampled_timesteps=state.metrics.sampled_episodes+problem_state.sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        )

        state = state.replace(
            evox_state=evox_state,
            metrics=workflow_metrics
        )

        return train_metrics, state
