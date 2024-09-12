import copy

import chex
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Self

from evorl.distributed import POP_AXIS_NAME
from evorl.metrics import MetricBase

from .workflow import Workflow


class ECWorkflowMetric(MetricBase):
    best_objective: chex.Array
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_timesteps_m: chex.Array = jnp.zeros((), dtype=jnp.float32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class TrainMetric(MetricBase):
    objectives: chex.Array


class ECWorkflow(Workflow):
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

    @classmethod
    def enable_jit(cls) -> None:
        pass

    @classmethod
    def enable_pmap(cls, axis_name) -> None:
        pass
