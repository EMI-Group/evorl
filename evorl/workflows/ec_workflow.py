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
from evorl.distributed import POP_AXIS_NAME, tree_unpmap, psum, split_key_to_devices
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
        return self.pmap_axis_name is not None

    def setup(self, key: chex.PRNGKey) -> State:
        key, evox_key = jax.random.split(key)
        evox_state = self._workflow.init(evox_key)
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

    def setup_multiple_device(self, state: State) -> State:
        self.pmap_axis_name = POP_AXIS_NAME
        self.devices = jax.local_devices()

        evox_state = self._workflow.enable_multi_devices(
            state.evox_state, self.pmap_axis_name)

        key = split_key_to_devices(state.key, self.devices)
        workflow_metrics = jax.device_put_replicated(
            state.metrics, self.devices)

        return state.replace(
            evox_state=evox_state,
            key=key,
            metrics=workflow_metrics
        )

    def step(self, state: State) -> Tuple[TrainMetric, State]:
        train_info, evox_state = self._workflow.step(state.evox_state)

        train_info = tree_unpmap(train_info, axis_name=self.pmap_axis_name)
        problem_state = state.evox_state.get_child_state('problem')

        # turn back to the original objectives
        train_metrics = TrainMetric(
            objectives=train_info['fitness'] * self._workflow.opt_direction
        )
        workflow_metrics = WorkflowMetric(
            sampled_episodes=state.metrics.sampled_episodes+problem_state.sampled_episodes,
            sampled_timesteps=state.metrics.sampled_timesteps+problem_state.sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        )

        state = state.replace(
            evox_state=evox_state,
            metrics=workflow_metrics
        )

        return train_metrics, state
