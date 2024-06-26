from functools import partial
from typing import Callable, Dict, List, Optional, Union, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import chex
import orbax.checkpoint as ocp

from evox import Algorithm, Problem, State, Monitor
from evox.workflows import StdWorkflow as EvoXWorkflow

from evorl.recorders import Recorder, ChainRecorder
from evorl.metrics import MetricBase
from evorl.utils.cfg_utils import get_output_dir
from .workflow import Workflow


class TrainMetric(MetricBase):
    objectives: chex.Array


class ECWorkflow(Workflow):
    def __init__(
        self,
        algorithm: Algorithm,
        problem: Union[Problem, List[Problem]],
        opt_direction: Union[str, List[str]] = 'max',
        candidate_transforms: List[Callable] = [],
        fitness_transforms: List[Callable] = [],
    ):
        self._workflow = EvoXWorkflow(
            algorithm=algorithm,
            problem=problem,
            monitors=[],
            opt_direction=opt_direction,
            candidate_transforms=candidate_transforms,
            fitness_transforms=fitness_transforms,
            jit_problem=True,
        )
        self.recorder = ChainRecorder([])  # dummy recorder

    def setup(self, key: chex.PRNGKey) -> State:
        return self._workflow.init(key)

    def step(self, state: State) -> Tuple[TrainMetric, State]:
        train_info, state = self._workflow.step(state)

        # turn back to the original objectives
        train_metric = TrainMetric(
            objectives=train_info['fitness'] * self._workflow.opt_direction
        )
        return train_metric, state
    
    def learn(self, state: State) -> State:
        pass

    def add_recorders(self, recorders: Recorder) -> None:
        for recorder in recorders:
            self.recorder.add_recorder(recorder)

    def close(self) -> None:
        self.recorder.close()