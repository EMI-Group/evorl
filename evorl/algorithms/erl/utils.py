import jax
import jax.tree_util as jtu

from evorl.sample_batch import SampleBatch


def flatten_pop_rollout_episode(trajectory: SampleBatch):
    """
    Flatten the trajectory from [#pop, T, B, ...] to [T, #pop*B, ...]
    """
    return jtu.tree_map(lambda x: jax.lax.collapse(x.swapaxes(0, 1), 1, 3), trajectory)
