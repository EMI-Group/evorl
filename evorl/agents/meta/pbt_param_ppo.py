import copy
import logging
import math
import numpy as np
import pandas as pd
from functools import partial


import chex
import hydra
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from optax.schedules import InjectStatefulHyperparamsState
import orbax.checkpoint as ocp

from jax.sharding import Mesh, NamedSharding, PositionalSharding
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

from omegaconf import DictConfig, OmegaConf, open_dict, read_write

from evorl.distributed import (
    POP_AXIS_NAME,
    tree_device_get,
    tree_device_put,
    parallel_map,
)
from evorl.metrics import MetricBase
from evorl.types import PyTreeData, PyTreeDict, State, MISSING_REWARD
from evorl.utils.jax_utils import tree_last
from evorl.workflows import RLWorkflow, Workflow
from evorl.metrics import WorkflowMetric

from .pbt import PBTWorkflow


class PBTParamPPOWorkflow(PBTWorkflow):
    @classmethod
    def name(cls):
        return "PBT-ParamPPO"

    def setup(self, key: chex.PRNGKey):
        pop_size = self.config.pop_size

        key, workflow_key, pop_key = jax.random.split(key, num=3)

        def _uniform_init(key, search_space):
            return jax.random.uniform(
                key,
                (pop_size,),
                minval=search_space.low,
                maxval=search_space.high,
            )

        search_space = self.config.search_space
        pop = PyTreeDict(
            {
                hp: _uniform_init(key, search_space[hp])
                for hp, key in zip(
                    search_space.keys(), jax.random.split(pop_key, len(search_space))
                )
            }
        )
        pop = tree_device_put(pop, self.sharding)

        # save metric on GPU0
        workflow_metrics = WorkflowMetric()

        workflow_keys = jax.random.split(workflow_key, pop_size)
        workflow_keys = jax.device_put(workflow_keys, self.sharding)

        pop_workflow_state = jax.jit(
            jax.vmap(self.workflow.setup, spmd_axis_name=POP_AXIS_NAME),
            in_shardings=self.sharding,
            out_shardings=self.sharding,
        )(workflow_keys)

        pop_workflow_state = jax.jit(
            jax.vmap(
                self.apply_hyperparams_to_workflow_state, spmd_axis_name=POP_AXIS_NAME
            ),
            in_shardings=self.sharding,
            out_shardings=self.sharding,
        )(pop, pop_workflow_state)

        return State(
            key=key,
            metrics=workflow_metrics,
            pop_workflow_state=pop_workflow_state,
            pop=pop,
        )

    def apply_hyperparams_to_workflow_state(
        self, hyperparams: PyTreeDict[str, chex.Numeric], workflow_state: State
    ):
        return workflow_state.replace(hp_state=hyperparams)
