from abc import ABCMeta, abstractmethod

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import PyTreeData, PyTreeNode
from evorl.utils.jax_utils import tree_get, tree_set


class ReplayBufferState(PyTreeData):
    """Contains data related to a replay buffer.

    Attributes:
        data: the stored replay buffer data.
        current_index: the pointer used for adding data.
        buffer_size: the current size of the replay buffer.
    """

    data: chex.ArrayTree
    current_index: chex.Array = jnp.zeros((), jnp.int32)
    buffer_size: chex.Array = jnp.zeros((), jnp.int32)


class AbstractReplayBuffer(PyTreeNode, metaclass=ABCMeta):
    """A ReplyBuffer Interface."""

    @abstractmethod
    def init(self, sample_spec: chex.ArrayTree) -> ReplayBufferState:
        """Initialize the state of the replay buffer.

        Args:
            sample_spec: A single sample or sample spec that contains the pytree structure and their dtype and shape.

        Returns:
            The initial state of the replay buffer.
        """
        pass

    @abstractmethod
    def add(
        self, buffer_state: ReplayBufferState, xs: chex.ArrayTree
    ) -> ReplayBufferState:
        """Add data to the replay buffer.

        Args:
            buffer_state: The current state of the replay buffer.
            xs: The data to add to the replay buffer.

        Returns:
            Updated state of the replay buffer.
        """
        pass

    @abstractmethod
    def sample(
        self, buffer_state: ReplayBufferState, key: chex.PRNGKey
    ) -> chex.ArrayTree:
        """Sample a batch of data from the replay buffer.

        Args:
            buffer_state: The current state of the replay buffer.
            key: JAX PRNGKey.

        Returns:
            A batch of data sampled from the replay buffer.
        """
        pass

    @abstractmethod
    def can_sample(self, buffer_state: ReplayBufferState) -> bool:
        """Check if the current replay buffer state can be used to sample.

        Args:
            buffer_state: The current state of the replay buffer.

        Returns:
            Whether the replay buffer is ready to call `sample()`.
        """
        pass

    @abstractmethod
    def is_full(self, buffer_state: ReplayBufferState) -> bool:
        pass


class ReplayBuffer(AbstractReplayBuffer):
    """ReplayBuffer with uniform sampling.

    Data are added and sampled in 1d-like structure.

    Attributes:
        capacity: the maximum capacity of the replay buffer.
        sample_batch_size: the batch size for `sample()`.
        min_sample_timesteps: the minimum number of timesteps before the replay buffer can sample.
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
            buffer_size=jnp.zeros((), jnp.int32),
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
        # Tips: when jit this function, set mask to static

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
