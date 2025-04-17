import logging
import math
from collections.abc import Sequence

import jax.tree_util as jtu
import numpy as np

from evorl.distributed import unpmap
from evorl.types import MISSING_REWARD, State
from evorl.utils.rl_toolkits import fold_multi_steps
from evorl.recorders import add_prefix

from evorl.algorithms.a2c import A2CWorkflow as _A2CWorkflow

logger = logging.getLogger(__name__)


class A2CWorkflow(_A2CWorkflow):
    @classmethod
    def name(cls):
        return "A2C-V2"

    def learn(self, state: State) -> State:
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        num_iters = math.ceil(self.config.total_timesteps / one_step_timesteps)

        steps_interval = self.config.eval_interval

        _multi_steps = fold_multi_steps(self.step, steps_interval)

        num_fold_iters = math.ceil(num_iters / steps_interval)

        for i in range(num_fold_iters):
            train_metrics_arr, state = _multi_steps(state)

            train_metrics_arr = unpmap(train_metrics_arr, self.pmap_axis_name)
            train_metrics = jtu.tree_map(lambda x: x[-1], train_metrics_arr)

            workflow_metrics = unpmap(state.metrics, self.pmap_axis_name)
            iterations = workflow_metrics.iterations.tolist()

            self.recorder.write(workflow_metrics.to_local_dict(), iterations)
            train_metric_data = train_metrics.to_local_dict()
            train_metric_data["train_episode_return"] = get_train_episode_return(
                train_metric_data["train_episode_return"]
            )
            self.recorder.write(train_metric_data, iterations)

            eval_metrics, state = self.evaluate(state)
            eval_metrics = unpmap(eval_metrics, self.pmap_axis_name)
            self.recorder.write(
                add_prefix(eval_metrics.to_local_dict(), "eval"), iterations
            )

            self.checkpoint_manager.save(
                iterations,
                unpmap(state, self.pmap_axis_name),
                force=i == num_fold_iters - 1,
            )

        return state


def _default_episode_return_reduce_fn(x):
    return x[-1]


def get_train_episode_return(
    episode_return_arr: Sequence[float], reduce_fn=_default_episode_return_reduce_fn
):
    """Handle episode return array with MISSING_REWARD, i.e., returned from multiple call of average_episode_discount_return."""
    episode_return_arr = np.array(episode_return_arr)
    mask = episode_return_arr == MISSING_REWARD
    if mask.all():
        return None
    else:
        return reduce_fn(episode_return_arr[~mask]).tolist()
