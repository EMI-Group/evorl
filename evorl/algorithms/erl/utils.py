import numpy as np

import jax
import jax.tree_util as jtu

from evorl.sample_batch import SampleBatch


def flatten_pop_rollout_episode(trajectory: SampleBatch):
    """
    Flatten the trajectory from [#pop, T, B, ...] to [T, #pop*B, ...]
    """
    return jtu.tree_map(lambda x: jax.lax.collapse(x.swapaxes(0, 1), 1, 3), trajectory)


def get_std_statistics(variance):
    def _get_stats(x):
        x = np.sqrt(x)
        return dict(
            min=np.min(x).tolist(),
            max=np.max(x).tolist(),
            mean=np.mean(x).tolist(),
        )

    return jtu.tree_map(_get_stats, variance)
