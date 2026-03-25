"""LAP (Loss-Adjusted Prioritization) Replay Buffer."""

import jax
import chex
import jax.numpy as jnp

from evorl.utils.jax_utils import tree_get
from .prioritized_replay_buffer import PrioritizedReplayBuffer, PrioritizedReplayBufferState



class LAPReplayBuffer(PrioritizedReplayBuffer):
    """ReplayBuffer with Loss-Adjusted Prioritization from TD7 paper.

    LAP is a variation of Prioritized Experience Replay (PER) that uses 
    proportional sampling but explicitly drops Importance Sampling (IS) weights,
    setting them uniformly to 1.0. 
    It leverages the parent `PrioritizedReplayBuffer` class for alpha-scaled
    priority sampling.

    Attributes:
        capacity: the maximum capacity of the replay buffer.
        sample_batch_size: the batch size for `sample()`.
        min_sample_timesteps: the minimum number of timesteps before sampling.
        alpha: the exponent determining how much prioritization is used (0 = uniform, 1 = full).
    """

    def sample(
        self,
        buffer_state: PrioritizedReplayBufferState,
        key: chex.PRNGKey,
        beta: float | chex.Array = 0.0,  # beta is ignored for LAP
    ) -> tuple[chex.ArrayTree, chex.Array, PrioritizedReplayBufferState]:
        """Sample a batch proportional to alpha-scaled priorities without IS weights.

        Args:
            buffer_state: Current buffer state.
            key: PRNG key.
            beta: Ignored mapping variable to match parent interface.

        Returns:
            A tuple of (batch, weights, updated_buffer_state). 
            The IS weights are uniformly set to 1.0.
        """
        # We can just call the parent sample with beta=0.0 which evaluates IS weights to 1.0.
        # However, to save math operations, we explicitly build the batch.

        # Mask out invalid priorities beyond buffer_size
        mask = jnp.arange(self.capacity) < buffer_state.buffer_size
        raw_priority = jnp.where(mask, buffer_state.priority, 0.0)
        
        # Apply alpha exponent
        priority_alpha = raw_priority ** self.alpha
        priority_alpha = jnp.where(mask, priority_alpha, 0.0)
        
        csum = jnp.cumsum(priority_alpha)
        val = jax.random.uniform(key, (self.sample_batch_size,)) * csum[-1]
        indices = jnp.searchsorted(csum, val)
        # Clamp indices to valid range
        indices = jnp.clip(indices, 0, buffer_state.buffer_size - 1)

        batch = tree_get(buffer_state.data, indices)
        
        # LAP explicitly ignores IS weights. Set to 1.0 uniformly.
        weights = jnp.ones(self.sample_batch_size, jnp.float32)

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
