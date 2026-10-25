from collections.abc import Callable

import jax
import jax.numpy as jnp
from gymnasium import spaces
from jax.custom_batching import custom_vmap
from jax.experimental import io_callback

from evorl.distributed.sharding import mark_varying


def sort_dict(d: dict) -> dict:
    return dict(sorted(d.items()))


def check_host_env_spaces(observation_space, action_space) -> None:
    """Check the spaces supported by the host environment adapters.

    Observations may be Box arrays, zero-based Discrete values or nested Dicts
    of these spaces. Actions may be floating-point Box arrays or zero-based
    Discrete values. Other structured and variable-length spaces are unsupported.

    Args:
        observation_space: Single-environment Gymnasium observation space.
        action_space: Single-environment Gymnasium action space.

    Raises:
        NotImplementedError: A space cannot be represented by this adapter.
    """

    def check_observation(space):
        if isinstance(space, spaces.Dict):
            for child in space.spaces.values():
                check_observation(child)
        elif not isinstance(space, (spaces.Box, spaces.Discrete)):
            raise NotImplementedError(
                f"Unsupported observation space: {space}. Expected Box, Discrete or Dict."
            )
        elif isinstance(space, spaces.Discrete) and space.start != 0:
            raise NotImplementedError(
                "The observation Discrete space must start at zero."
            )

    check_observation(observation_space)
    if not isinstance(action_space, (spaces.Box, spaces.Discrete)):
        raise NotImplementedError(
            f"Unsupported action space: {action_space}. Expected Box or Discrete."
        )
    if isinstance(action_space, spaces.Discrete) and action_space.start != 0:
        raise NotImplementedError("The action Discrete space must start at zero.")
    if isinstance(action_space, spaces.Box) and action_space.dtype.kind != "f":
        raise NotImplementedError("Box actions must have a floating-point dtype.")


def make_batched_io_callback(callback: Callable, result_spec) -> Callable:
    """Create an ordered host callback that merges nested `vmap` batches.

    The callback receives arrays with any mapped axes prepended to their
    ordinary input shape. It must process the whole batch in one call, and
    return outputs with the same mapped axes prepended to `result_spec`.
    Each batching rule prepends its mapped axis to the output specification.
    Custom batching handles these axes before the ordered IO callback is bound;
    JAX's default batching rule rejects ordered IO callbacks.
    Ordering also covers consecutive resets of the same native pool: a reset
    does not consume the preceding episode's callback token.
    Within `shard_map`, outputs are marked varying over the active Manual axes:
    separate host environment calls need not return identical values, even with
    identical inputs. This type cast performs no communication.

    Args:
        callback: Stateful host function accepting arrays.
        result_spec: Output PyTree of `jax.ShapeDtypeStruct` for an unmapped call.

    Returns:
        Callback that preserves host-call ordering and runs once per full batch.
    """

    @custom_vmap
    def call(*args):
        result = io_callback(callback, result_spec, *args, ordered=True)
        return mark_varying(result, jax.sharding.get_abstract_mesh().manual_axes)

    @call.def_vmap
    def batch(axis_size, in_batched, *args):
        args = tuple(
            x if mapped else jnp.broadcast_to(x, (axis_size, *x.shape))
            for x, mapped in zip(args, in_batched, strict=True)
        )
        batched_spec = jax.tree.map(
            lambda spec: jax.ShapeDtypeStruct((axis_size, *spec.shape), spec.dtype),
            result_spec,
        )
        result = make_batched_io_callback(callback, batched_spec)(*args)
        return result, jax.tree.map(lambda _: True, result)

    return call
