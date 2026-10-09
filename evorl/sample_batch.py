import jax
import chex

from .types import PyTreeData, PyTreeArrayMixin, ExtraInfo, Reward, RewardDict
from .utils.jax_utils import right_shift_with_padding


class SampleBatch(PyTreeData, PyTreeArrayMixin):
    """Data container for trajectory data."""

    obs: chex.ArrayTree | None = None
    actions: chex.ArrayTree | None = None
    rewards: Reward | RewardDict | None = None
    next_obs: jax.Array | None = None
    dones: jax.Array | None = None
    extras: ExtraInfo | None = None


class Episode(PyTreeData):
    """The container for an episode trajectory."""

    trajectory: SampleBatch

    @property
    def valid_mask(self) -> jax.Array:
        return 1 - right_shift_with_padding(self.trajectory.dones, 1)
