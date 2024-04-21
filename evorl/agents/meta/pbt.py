import jax
import jax.numpy as jnp
from flax import struct
import math

import hydra
from omegaconf import DictConfig


from evorl.sample_batch import SampleBatch
from evorl.networks import make_policy_network, make_value_network
from evorl.utils import running_statistics
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.toolkits import (
    compute_gae, flatten_rollout_trajectory,
    average_episode_discount_return
)
from evorl.workflows import OnPolicyRLWorkflow, RLWorkflow
from evorl.agents import AgentState
from evorl.distributed import agent_gradient_update, tree_unpmap, psum
from evorl.envs import create_env, Env, EnvState
from evorl.evaluator import Evaluator
from evorl.metrics import MetricBase
from ..agent import Agent, AgentState

from evox import State
# from evorl.types import State


import orbax.checkpoint as ocp
import chex
import optax
from evorl.types import (
    LossDict, Action, Params, PolicyExtraInfo, PyTreeDict, pytree_field,
    MISSING_REWARD
)
from evorl.metrics import TrainMetric, WorkflowMetric
from typing import Tuple, Sequence, Optional, Any
import logging
import flax.linen as nn
from flax import struct
import copy
from omegaconf import OmegaConf


class TrainMetric(MetricBase):
    best_workflow_return: chex.Array = jnp.zeros((), dtype=jnp.float32)


class PBTWorkflow(RLWorkflow):
    def __init__(self,
                 workflow: RLWorkflow,
                 config: DictConfig):
        super(PBTWorkflow, self).__init__(config)

        self.workflow = workflow

    @classmethod
    def build_from_config(cls, config: DictConfig, enable_jit: bool = True):
        config = copy.deepcopy(config)  # avoid in-place modification
        if devices is None:
            devices = jax.local_devices()

        # Tips: Multi-Device Training is done by sub-workflow
        if enable_jit:
            cls.enable_jit()

        OmegaConf.set_readonly(config, True)

        workflow = cls._build_from_config(config)

        return workflow

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        target_workflow_config = config.target_workflow
        target_workflow_cls = hydra.utils.get_class(
            target_workflow_config.workflow_cls)

        devices = jax.local_devices()
        if len(devices) > 1:
            target_workflow = target_workflow_cls.build_from_config(
                config, enable_multi_devices=True, devices=devices,
            )
        else:
            target_workflow = target_workflow_cls.build_from_config(
                config,
                enable_jit=True
            )

        return cls(target_workflow, config)

    def setup(self, key: chex.PRNGKey):
        pop_size = self.config.pop_size

        key, workflow_key, pop_key = jax.random.split(key, num=3)
        self.workflow.optimizer = optax.inject_hyperparams(
            optax.adam, static_args=('b1','b2','eps','eps_root')
        )(learning_rate=self.config.search_space.lr.low)

        pop_workflow_state = jax.vmap(self.workflow.setup)(
            jax.random.split(key, pop_size)
        )

        pop_lr = jax.random.uniform(pop_key, (pop_size,),
                                 minval=self.config.search_space.lr.low,
                                 maxval=self.config.search_space.lr.high)

        def _apply_lr_to_workflow_state(lr, workflow_state):
            opt_state = workflow_state.opt_state
            opt_state.hyperparams['learning_rate'] = lr
            return workflow_state.update(opt_state=opt_state)
        
        pop_workflow_state = jax.vmap(_apply_lr_to_workflow_state)(
            pop_lr, pop_workflow_state
        )

        workflow_metrics = self._setup_workflow_metrics()

        return State(
            key=key,
            metrics=workflow_metrics,
            pop_workflow_state=pop_workflow_state,
            pop_lr=pop_lr,
        )
    
    def step(self, state: State) -> Tuple[TrainMetric, State]:
        pop_workflow_state = state.pop_workflow_state
        pop_lr = state.pop_lr

        pop_workflow_state, metrics = jax.vmap(self.workflow.step)(
            pop_workflow_state
        )

        return metrics, state.update(pop_workflow_state=pop_workflow_state)
        


    def learn(self, state: State) -> State:
        for _ in range(self.config.num_steps):
            metrics, state = self.step(state)
            self.recorder.record(metrics)

        return state