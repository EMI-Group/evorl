from abc import ABCMeta, abstractmethod

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import PyTreeData, PyTreeNode
from evorl.utils.jax_utils import tree_get, tree_set


class ReplayBufferState(PyTreeData):
    """Contains data related to a replay buffer."""

    data: chex.ArrayTree
    current_index: chex.Array = jnp.zeros((), jnp.int32)
    buffer_size: chex.Array = jnp.zeros((), jnp.uint32)


class AbstractReplayBuffer(PyTreeNode, metaclass=ABCMeta):
    @abstractmethod
    def init(self, sample_spec: chex.ArrayTree):
        """
        Args:
            sample_spec: a single sample that contains the pytree structure and their dtype and shape
        """
        pass

    @abstractmethod
    def add(
        self, buffer_state: ReplayBufferState, xs: chex.ArrayTree
    ) -> ReplayBufferState:
        pass

    @abstractmethod
    def sample(
        self, buffer_state: ReplayBufferState, key: chex.PRNGKey
    ) -> chex.ArrayTree:
        pass

    @abstractmethod
    def can_sample(self, buffer_state: ReplayBufferState) -> bool:
        pass

    @abstractmethod
    def is_full(self, buffer_state: ReplayBufferState) -> bool:
        pass


class ReplayBuffer(AbstractReplayBuffer):
    """
    ReplayBuffer with uniform sampling. Data are added and sampled in 1d-like structure.
    """

    capacity: int
    sample_batch_size: int
    min_sample_timesteps: int = 0

    def init(self, spec: chex.ArrayTree) -> ReplayBufferState:
        # Note: broadcast_to will not pre-allocate memory
        data = jtu.tree_map(
            lambda x: jnp.broadcast_to(jnp.empty_like(x), (self.capacity, *x.shape)),
            spec,
        )

        return ReplayBufferState(
            data=data,
            current_index=jnp.zeros((), jnp.int32),
            buffer_size=jnp.zeros((), jnp.uint32),
        )

    def is_full(self, buffer_state: ReplayBufferState) -> bool:
        return buffer_state.buffer_size == self.capacity

    def can_sample(self, buffer_state: ReplayBufferState) -> bool:
        return buffer_state.buffer_size >= self.min_sample_timesteps

    def add(
        self,
        buffer_state: ReplayBufferState,
        xs: chex.ArrayTree,
        mask: chex.Array | None = None,
    ) -> ReplayBufferState:
        """
        Tips: when jit this function, set mask to static
        """

        chex.assert_trees_all_equal_dtypes(xs, buffer_state.data)

        if mask is not None:
            assert mask.ndim == 1
            chex.assert_tree_shape_prefix(xs, mask.shape)
            batch_size = mask.sum()

            # Note: here we utilize the feature of jax.Array with mode="promise_in_bounds",
            # that indices on [self.capacity] will be ignore when call set()
            # eg: mask = [1,0,1,1,0], capacity = n > 5
            # Then, cumsum_mask = [1,1,2,3,3], cumsum_mask-1 = [0,0,1,2,2]
            # assume current_index = 0, then indices = [0,n,1,2,n]
            cumsum_mask = jnp.cumsum(mask, axis=0, dtype=jnp.int32)
            indices = (buffer_state.current_index + cumsum_mask - 1) % self.capacity
            indices = jnp.where(mask, indices, self.capacity)
        else:
            batch_size = jtu.tree_leaves(xs)[0].shape[0]

            indices = (
                buffer_state.current_index + jnp.arange(batch_size, dtype=jnp.int32)
            ) % self.capacity

        data = tree_set(buffer_state.data, xs, indices, unique_indices=False)

        current_index = (buffer_state.current_index + batch_size) % self.capacity
        buffer_size = jnp.minimum(buffer_state.buffer_size + batch_size, self.capacity)

        return buffer_state.replace(
            data=data, current_index=current_index, buffer_size=buffer_size
        )

    def sample(
        self, buffer_state: ReplayBufferState, key: chex.ArrayTree
    ) -> chex.ArrayTree:
        indices = jax.random.randint(
            key, (self.sample_batch_size,), minval=0, maxval=buffer_state.buffer_size
        )

        batch = tree_get(buffer_state.data, indices)

        return batch
