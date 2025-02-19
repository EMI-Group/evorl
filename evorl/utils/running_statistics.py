import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import PyTreeData

from .jax_utils import tree_ones_like, tree_zeros_like

"""Utility functions to compute running statistics.

Modified from https://github.com/google/brax/blob/main/brax/training/acme/running_statistics.py
"""


class NestedMeanStd(PyTreeData):
    """A container for running statistics (mean, std) of possibly nested data."""

    mean: chex.ArrayTree
    std: chex.ArrayTree


class RunningStatisticsState(NestedMeanStd):
    """Full state of running statistics computation."""

    count: chex.Array
    summed_variance: chex.ArrayTree


def init_state(
    nest: chex.ArrayTree, int_counter: bool = False
) -> RunningStatisticsState:
    """Initializes the running statistics for the given nested structure."""
    dtype_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
    dtype_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    return RunningStatisticsState(
        count=jnp.zeros((), dtype=dtype_int if int_counter else dtype_float),
        mean=tree_zeros_like(nest, dtype=dtype_float),
        summed_variance=tree_zeros_like(nest, dtype=dtype_float),
        # Initialize with ones to make sure normalization works correctly
        # in the initial state.
        std=tree_ones_like(nest, dtype=dtype_float),
    )


def _validate_batch_shapes(
    batch: chex.Array, reference_sample: chex.Array, batch_dims: tuple[int, ...]
) -> None:
    """Verifies shapes of the batch leaves against the reference sample.

    Checks that batch dimensions are the same in all leaves in the batch.
    Checks that non-batch dimensions for all leaves in the batch are the same
    as in the reference sample.

    Arguments:
      batch: the nested batch of data to be verified.
      reference_sample: the nested array to check non-batch dimensions.
      batch_dims: a Tuple of indices of batch dimensions in the batch shape.

    Returns:
      None.
    """

    def validate_node_shape(reference_sample: chex.Array, batch: chex.Array) -> None:
        expected_shape = batch_dims + reference_sample.shape
        # assert batch.shape == expected_shape, f'{batch.shape} != {expected_shape}'
        chex.assert_shape(
            batch, expected_shape, custom_message=f"{batch.shape} != {expected_shape}"
        )

    jtu.tree_map(validate_node_shape, reference_sample, batch)


def update(
    state: RunningStatisticsState,
    batch: chex.ArrayTree,
    *,
    weights: chex.Array | None = None,
    std_min_value: float = 1e-6,
    std_max_value: float = 1e6,
    pmap_axis_name: str | None = None,
    validate_shapes: bool = True,
) -> RunningStatisticsState:
    """Updates the running statistics with the given batch of data.

    Note: data batch and state elements (mean, etc.) must have the same structure.

    Note: by default will use int32 for counts and float32 for accumulated
    variance. This results in an integer overflow after 2^31 data points and
    degrading precision after 2^24 batch updates or even earlier if variance
    updates have large dynamic range.
    To improve precision, consider setting jax_enable_x64 to True, see
    https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision

    Arguments:
      state: The running statistics before the update.
      batch: The data to be used to update the running statistics.
      weights: Weights of the batch data. Should match the batch dimensions.
        Passing a weight of 2. should be equivalent to updating on the
        corresponding data point twice.
      std_min_value: Minimum value for the standard deviation.
      std_max_value: Maximum value for the standard deviation.
      pmap_axis_name: Name of the pmapped axis, if any.
      validate_shapes: If true, the shapes of all leaves of the batch will be
        validated. Enabled by default. Doesn't impact performance when jitted.

    Returns:
      Updated running statistics.
    """
    # We require exactly the same structure to avoid issues when flattened
    # batch and state have different order of elements.
    assert jtu.tree_structure(batch) == jtu.tree_structure(state.mean)
    batch_shape = jtu.tree_leaves(batch)[0].shape
    # We assume the batch dimensions always go first.
    batch_dims = batch_shape[: len(batch_shape) - jtu.tree_leaves(state.mean)[0].ndim]
    batch_axis = range(len(batch_dims))
    if weights is None:
        step_increment = jnp.prod(jnp.array(batch_dims))
    else:
        step_increment = jnp.sum(weights)
    if pmap_axis_name is not None:
        step_increment = jax.lax.psum(step_increment, axis_name=pmap_axis_name)
    count = state.count + step_increment

    # Validation is important. If the shapes don't match exactly, but are
    # compatible, arrays will be silently broadcasted resulting in incorrect
    # statistics.
    if validate_shapes:
        if weights is not None:
            if weights.shape != batch_dims:
                raise ValueError(f"{weights.shape} != {batch_dims}")
        _validate_batch_shapes(batch, state.mean, batch_dims)

    def _compute_node_statistics(
        mean: chex.Array, summed_variance: chex.Array, batch: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        assert isinstance(mean, chex.Array), type(mean)
        assert isinstance(summed_variance, chex.Array), type(summed_variance)
        # The mean and the sum of past variances are updated with Welford's
        # algorithm using batches (see https://stackoverflow.com/q/56402955).
        diff_to_old_mean = batch - mean
        if weights is not None:
            expanded_weights = jnp.reshape(
                weights, list(weights.shape) + [1] * (batch.ndim - weights.ndim)
            )
            diff_to_old_mean = diff_to_old_mean * expanded_weights
        mean_update = jnp.sum(diff_to_old_mean, axis=batch_axis) / count
        if pmap_axis_name is not None:
            mean_update = jax.lax.psum(mean_update, axis_name=pmap_axis_name)
        mean = mean + mean_update

        diff_to_new_mean = batch - mean
        variance_update = diff_to_old_mean * diff_to_new_mean
        variance_update = jnp.sum(variance_update, axis=batch_axis)
        if pmap_axis_name is not None:
            variance_update = jax.lax.psum(variance_update, axis_name=pmap_axis_name)
        summed_variance = summed_variance + variance_update
        return mean, summed_variance

    updated_stats = jtu.tree_map(
        _compute_node_statistics, state.mean, state.summed_variance, batch
    )
    # Extract `mean` and `summed_variance` from `updated_stats` nest.
    mean = jtu.tree_map(lambda _, x: x[0], state.mean, updated_stats)
    summed_variance = jtu.tree_map(lambda _, x: x[1], state.mean, updated_stats)

    def compute_std(summed_variance: chex.Array, std: chex.Array) -> chex.Array:
        assert isinstance(summed_variance, chex.Array)
        # Summed variance can get negative due to rounding errors.
        summed_variance = jnp.maximum(summed_variance, 0)
        std = jnp.sqrt(summed_variance / count)
        std = jnp.clip(std, std_min_value, std_max_value)
        return std

    std = jtu.tree_map(compute_std, summed_variance, state.std)

    return RunningStatisticsState(
        count=count, mean=mean, summed_variance=summed_variance, std=std
    )


def normalize(
    batch: chex.Array,
    mean_std: NestedMeanStd,
    eps: float = 1e-8,
    max_abs_value: float | None = None,
) -> chex.Array:
    """Normalizes data using running statistics."""

    def normalize_leaf(
        data: chex.Array, mean: chex.Array, std: chex.Array
    ) -> chex.Array:
        # Only normalize inexact
        if not jnp.issubdtype(data.dtype, jnp.inexact):
            return data
        data = (data - mean) / (std + eps)
        if max_abs_value is not None:
            # TODO: remove pylint directive
            data = jnp.clip(data, -max_abs_value, +max_abs_value)
        return data

    return jtu.tree_map(normalize_leaf, batch, mean_std.mean, mean_std.std)


def denormalize(batch: chex.Array, mean_std: NestedMeanStd) -> chex.Array:
    """Denormalizes values in a nested structure using the given mean/std.

    Only values of inexact types are denormalized.
    See https://numpy.org/doc/stable/_images/dtype-hierarchy.png for Numpy type
    hierarchy.

    Args:
      batch: a nested structure containing batch of data.
      mean_std: mean and standard deviation used for denormalization.

    Returns:
      Nested structure with denormalized values.
    """

    def denormalize_leaf(
        data: chex.Array, mean: chex.Array, std: chex.Array
    ) -> chex.Array:
        # Only denormalize inexact
        if not jnp.issubdtype(data.dtype, jnp.inexact):
            return data
        return data * std + mean

    return jtu.tree_map(denormalize_leaf, batch, mean_std.mean, mean_std.std)
