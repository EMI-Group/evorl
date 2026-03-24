"""Prioritized Replay Buffer with LAP (Loss-Adjusted Prioritization) support."""

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.utils.jax_utils import tree_get

from .replay_buffer import ReplayBuffer, ReplayBufferState


class PrioritizedReplayBufferState(ReplayBufferState):
    """State for the prioritized replay buffer.

    Attributes:
        priority: Priority values for each entry in the buffer.
        max_priority: Current maximum priority value.
        sample_indices: Indices of the last sampled batch (for priority updates).
    """

    priority: chex.Array = jnp.zeros((), jnp.float32)
    max_priority: chex.Array = jnp.ones((), jnp.float32)
    sample_indices: chex.Array = jnp.zeros((), jnp.int32)


class PrioritizedReplayBuffer(ReplayBuffer):
    """ReplayBuffer with proportional prioritized sampling (LAP).

    Uses Loss-Adjusted Prioritization from TD7 paper.
    New samples are assigned max_priority. Priorities are updated
    based on TD-error after critic training.

    Attributes:
        capacity: the maximum capacity of the replay buffer.
        sample_batch_size: the batch size for `sample()`.
        min_sample_timesteps: the minimum number of timesteps before the
            replay buffer can sample.
        alpha: the exponent determining how much prioritization is used (0 = uniform, 1 = full).
    """

    alpha: float = 0.6

    def init(self, spec: chex.ArrayTree) -> PrioritizedReplayBufferState:
        data = jtu.tree_map(
            lambda x: jnp.broadcast_to(jnp.empty_like(x), (self.capacity, *x.shape)),
            spec,
        )

        return PrioritizedReplayBufferState(
            data=data,
            current_index=jnp.zeros((), jnp.int32),
            buffer_size=jnp.zeros((), jnp.int32),
            priority=jnp.zeros(self.capacity, jnp.float32),
            max_priority=jnp.ones((), jnp.float32),
            sample_indices=jnp.zeros(self.sample_batch_size, jnp.int32),
        )

    def add(
        self,
        buffer_state: PrioritizedReplayBufferState,
        xs: chex.ArrayTree,
        mask: chex.Array | None = None,
    ) -> PrioritizedReplayBufferState:
        # Get indices before calling parent add
        if mask is not None:
            batch_size = mask.sum()
            cumsum_mask = jnp.cumsum(mask, axis=0, dtype=jnp.int32)
            indices = (buffer_state.current_index + cumsum_mask - 1) % self.capacity
            indices = jnp.where(mask, indices, self.capacity)
        else:
            batch_size = jtu.tree_leaves(xs)[0].shape[0]
            indices = (
                buffer_state.current_index + jnp.arange(batch_size, dtype=jnp.int32)
            ) % self.capacity

        # Call parent add for data
        new_state = super().add(buffer_state, xs, mask)

        # Set priorities for new entries to max_priority
        priority = buffer_state.priority.at[indices].set(
            buffer_state.max_priority,
            mode="drop",
        )

        return PrioritizedReplayBufferState(
            data=new_state.data,
            current_index=new_state.current_index,
            buffer_size=new_state.buffer_size,
            priority=priority,
            max_priority=buffer_state.max_priority,
            sample_indices=buffer_state.sample_indices,
        )

    def sample(
        self,
        buffer_state: PrioritizedReplayBufferState,
        key: chex.PRNGKey,
        beta: float | chex.Array = 0.4,
    ) -> tuple[chex.ArrayTree, chex.Array, PrioritizedReplayBufferState]:
        """Sample a batch proportional to priorities with IS weights.

        Args:
            buffer_state: Current buffer state.
            key: PRNG key.
            beta: Importance sampling exponent.

        Returns:
            A tuple of (batch, weights, updated_buffer_state). 
            The weights are the computed Importance Sampling (IS) weights.
        """
        # Mask out invalid priorities beyond buffer_size
        mask = jnp.arange(self.capacity) < buffer_state.buffer_size
        raw_priority = jnp.where(mask, buffer_state.priority, 0.0)
        
        # Apply alpha exponent
        priority_alpha = raw_priority ** self.alpha
        priority_alpha = jnp.where(mask, priority_alpha, 0.0)
        
        sum_priority = jnp.sum(priority_alpha)
        # Compute probabilities P(i) = p_i^alpha / sum(p_i^alpha)
        # We avoid division here and directly use priority_alpha for csum
        
        csum = jnp.cumsum(priority_alpha)
        val = jax.random.uniform(key, (self.sample_batch_size,)) * csum[-1]
        indices = jnp.searchsorted(csum, val)
        # Clamp indices to valid range
        indices = jnp.clip(indices, 0, buffer_state.buffer_size - 1)

        batch = tree_get(buffer_state.data, indices)
        
        # Compute IS weights: w_i = (N * P(i)) ** -beta / max(w_i)
        # P(i) = priority_alpha[indices] / sum_priority
        N = jnp.maximum(1, buffer_state.buffer_size)
        p_i = priority_alpha[indices] / sum_priority
        weights = (N * p_i) ** (-beta)
        # Normalize weights by max weight in batch
        weights = weights / jnp.max(weights)

        # Store indices for later priority update
        new_state = PrioritizedReplayBufferState(
            data=buffer_state.data,
            current_index=buffer_state.current_index,
            buffer_size=buffer_state.buffer_size,
            priority=buffer_state.priority,
            max_priority=buffer_state.max_priority,
            sample_indices=indices,
        )

        return batch, weights, new_state

    def update_priority(
        self,
        buffer_state: PrioritizedReplayBufferState,
        priority: chex.Array,
    ) -> PrioritizedReplayBufferState:
        """Update priorities at the last sampled indices.

        Args:
            buffer_state: Current buffer state (must have valid sample_indices).
            priority: New priority values, shape (sample_batch_size,).

        Returns:
            Updated buffer state with new priorities.
        """
        new_priority = buffer_state.priority.at[buffer_state.sample_indices].set(
            priority
        )
        new_max_priority = jnp.maximum(buffer_state.max_priority, priority.max())

        return PrioritizedReplayBufferState(
            data=buffer_state.data,
            current_index=buffer_state.current_index,
            buffer_size=buffer_state.buffer_size,
            priority=new_priority,
            max_priority=new_max_priority,
            sample_indices=buffer_state.sample_indices,
        )

    def reset_max_priority(
        self, buffer_state: PrioritizedReplayBufferState
    ) -> PrioritizedReplayBufferState:
        """Recompute max_priority from current buffer entries."""
        mask = jnp.arange(self.capacity) < buffer_state.buffer_size
        valid_priority = jnp.where(mask, buffer_state.priority, -jnp.inf)
        # Handle empty buffer case by defaulting to 1.0 (or current max)
        new_max_priority = jnp.maximum(valid_priority.max(), 1e-5)

        return PrioritizedReplayBufferState(
            data=buffer_state.data,
            current_index=buffer_state.current_index,
            buffer_size=buffer_state.buffer_size,
            priority=buffer_state.priority,
            max_priority=new_max_priority,
            sample_indices=buffer_state.sample_indices,
        )
