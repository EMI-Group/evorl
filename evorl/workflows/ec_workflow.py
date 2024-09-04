import copy
from collections.abc import Callable, Sequence

import chex
import jax
import jax.numpy as jnp
from evox import Algorithm, Problem
from evox.workflows import StdWorkflow as EvoXWorkflow
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Self

from evorl.agent import Agent
from evorl.distributed import (
    POP_AXIS_NAME,
    get_global_ranks,
    psum,
    split_key_to_devices,
)
from evorl.metrics import MetricBase
from evorl.types import State

from .workflow import Workflow


class ECWorkflowMetric(MetricBase):
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_timesteps_m: chex.Array = jnp.zeros((), dtype=jnp.float32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class TrainMetric(MetricBase):
    objectives: chex.Array


class ECWorkflow(Workflow):
    def __init__(
        self,
        config: DictConfig,
        agent: Agent,
        algorithm: Algorithm,
        problem: Problem,
        opt_direction: str | Sequence[str] = "max",
        candidate_transforms: Sequence[Callable] = (),
        fitness_transforms: Sequence[Callable] = (),
    ):
        super().__init__(config)

        self.agent = agent
        self._workflow = EvoXWorkflow(
            algorithm=algorithm,
            problem=problem,
            monitors=[],
            opt_direction=opt_direction,
            candidate_transforms=candidate_transforms,
            fitness_transforms=fitness_transforms,
            jit_step=False,  # don't jit internally
        )
        self.pmap_axis_name = None
        self.devices = jax.local_devices()[:1]

    @property
    def enable_multi_devices(self) -> bool:
        return self.pmap_axis_name is not None

    @classmethod
    def build_from_config(
        cls,
        config: DictConfig,
        enable_multi_devices: bool = False,
        enable_jit: bool = True,
    ):
        config = copy.deepcopy(config)  # avoid in-place modification

        devices = jax.local_devices()

        if enable_multi_devices:
            cls.enable_pmap(POP_AXIS_NAME)
            OmegaConf.set_readonly(config, False)
            cls._rescale_config(config)
        elif enable_jit:
            cls.enable_jit()

        OmegaConf.set_readonly(config, True)

        workflow = cls._build_from_config(config)
        if enable_multi_devices:
            workflow.pmap_axis_name = POP_AXIS_NAME
            workflow.devices = devices

        return workflow

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        raise NotImplementedError

    @staticmethod
    def _rescale_config(config: DictConfig) -> None:
        """
        When enable_multi_devices=True, rescale config settings in-place to match multi-devices.
        Note: not need for EvoX part, as it's already handled by EvoX.
        """
        pass

    def _setup_workflow_metrics(self) -> MetricBase:
        """
        Customize the workflow metrics.
        """
        return ECWorkflowMetric()

    def setup(self, key: chex.PRNGKey) -> State:
        key, evox_key = jax.random.split(key, 2)
        evox_state = self._workflow.init(evox_key)
        workflow_metrics = self._setup_workflow_metrics()

        if self.enable_multi_devices:
            # Note: we don't use evox's enable_multi_devices(),
            # instead we use our own implementation
            self._workflow.pmap_axis_name = self.pmap_axis_name
            self._workflow.devices = self.devices

            evox_state, workflow_metrics = jax.device_put_replicated(
                (evox_state, workflow_metrics), self.devices
            )
            key = split_key_to_devices(key, self.devices)
            evox_state = evox_state.replace(
                rank=get_global_ranks(), world_size=jax.device_count()
            )

        return State(key=key, evox_state=evox_state, metrics=workflow_metrics)

    def step(self, state: State) -> tuple[TrainMetric, State]:
        train_info, evox_state = self._workflow.step(state.evox_state)

        problem_state = state.evox_state.get_child_state("problem")
        sampled_episodes = psum(problem_state.sampled_episodes, self.pmap_axis_name)
        sampled_timesteps_m = (
            psum(problem_state.sampled_timesteps, self.pmap_axis_name) / 1e6
        )
        # turn back to the original objectives
        # Note: train_info['fitness'] is already all-gathered in evox
        train_metrics = TrainMetric(
            objectives=train_info["fitness"] * self._workflow.opt_direction
        )

        workflow_metrics = ECWorkflowMetric(
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
            sampled_timesteps_m=state.metrics.sampled_timesteps_m + sampled_timesteps_m,
            iterations=state.metrics.iterations + 1,
        )

        state = state.replace(evox_state=evox_state, metrics=workflow_metrics)

        return train_metrics, state

    @classmethod
    def enable_jit(cls) -> None:
        cls.step = jax.jit(cls.step, static_argnums=(0,))

    @classmethod
    def enable_pmap(cls, axis_name) -> None:
        cls.step = jax.pmap(cls.step, axis_name, static_broadcasted_argnums=(0,))
