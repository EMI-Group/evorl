import logging
from functools import partial
import math
from typing_extensions import Self  # pytype: disable=not-supported-yet]
from omegaconf import DictConfig

import chex
import jax
import jax.tree_util as jtu

from evorl.distributed import (
    unpmap,
)
from evorl.types import State, MISSING_REWARD
from evorl.metrics import MetricBase
from evorl.utils.jax_utils import scan_and_last
from evorl.recorders import add_prefix, get_1d_array_statistics, get_1d_array

from evorl.algorithms.ppo import PPOWorkflow

logger = logging.getLogger(__name__)


class PopPPOWorkflow(PPOWorkflow):
    @classmethod
    def name(cls):
        return "PopPPO"

    @classmethod
    def build_from_config(
        cls,
        config: DictConfig,
        enable_multi_devices: bool = False,
        enable_jit: bool = True,
    ) -> Self:
        devices = jax.local_devices()

        if enable_multi_devices or len(devices) > 1:
            raise NotImplementedError("Multi-devices is not supported yet.")

        return super().build_from_config(config, enable_multi_devices, enable_jit)

    def setup(self, key: chex.PRNGKey) -> State:
        state = jax.vmap(super().setup)(
            jax.random.split(key, self.config.pop_size),
        )

        return state

    def evaluate(self, state):
        return jax.vmap(super().evaluate)(state)

    def step(self, state: State) -> tuple[MetricBase, State]:
        return jax.vmap(super().step)(state)

    def _multi_steps(self, state):
        def _step(state, _):
            train_metrics, state = self.step(state)
            return state, train_metrics

        state, train_metrics = scan_and_last(
            _step, state, (), length=self.config.fold_iters
        )

        return train_metrics, state

    def learn(self, state: State) -> State:
        one_step_timesteps = (
            self.config.rollout_length * self.config.num_envs * self.config.fold_iters
        )
        sampled_timesteps = unpmap(state.metrics.sampled_timesteps).tolist()[0]
        num_iters = math.ceil(
            (self.config.total_timesteps - sampled_timesteps) / one_step_timesteps
        )

        for i in range(num_iters):
            train_metrics, state = self._multi_steps(state)
            workflow_metrics = state.metrics

            iters = unpmap(state.metrics.iterations, self.pmap_axis_name).tolist()[0]
            train_metrics = unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = unpmap(workflow_metrics, self.pmap_axis_name)

            workflow_metrics_data = jtu.tree_map(
                lambda x: x[0],
                workflow_metrics.to_local_dict(),
            )

            self.recorder.write(workflow_metrics_data, iters)

            train_metric_data = train_metrics.to_local_dict()
            train_episode_return = train_metric_data["train_episode_return"]
            train_episode_return = train_episode_return[
                train_episode_return != MISSING_REWARD
            ]
            if len(train_episode_return) > 0:
                train_metric_data["train_episode_return"] = train_episode_return
            else:
                train_metric_data["train_episode_return"] = None

            train_metric_data = jtu.tree_map(
                partial(get_1d_array_statistics, histogram=True),
                train_metric_data,
            )
            self.recorder.write(train_metric_data, iters)

            if iters % self.config.eval_interval == 0 or iters == num_iters:
                eval_metrics, state = self.evaluate(state)
                eval_metrics = unpmap(eval_metrics, self.pmap_axis_name)

                eval_metrics_dict = jtu.tree_map(
                    get_1d_array,
                    eval_metrics.to_local_dict(),
                )

                self.recorder.write(add_prefix(eval_metrics_dict, "eval"), iters)

            self.checkpoint_manager.save(
                iters,
                unpmap(state, self.pmap_axis_name),
                force=iters == num_iters,
            )

        return state
