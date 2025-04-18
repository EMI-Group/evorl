import chex

from .types import PyTreeData, PyTreeArrayMixin, ExtraInfo, Reward, RewardDict
from .utils.jax_utils import right_shift_with_padding


class SampleBatch(PyTreeData, PyTreeArrayMixin):
    """Data container for trajectory data."""

    obs: chex.ArrayTree | None = None
    actions: chex.ArrayTree | None = None
    rewards: Reward | RewardDict | None = None
    next_obs: chex.Array | None = None
    dones: chex.Array | None = None
    extras: ExtraInfo | None = None


class Episode(PyTreeData):
    """The container for an episode trajectory."""

    trajectory: SampleBatch

    @property
    def valid_mask(self) -> chex.Array:
        return 1 - right_shift_with_padding(self.trajectory.dones, 1)
