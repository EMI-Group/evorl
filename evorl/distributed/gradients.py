from collections.abc import Callable

import jax
import optax
from flax import nnx

"""Training gradient utility functions.

Modified from https://github.com/google/brax/blob/main/brax/training/gradients.py
"""


def pmean_loss_and_grad(
    loss_fn: Callable, axis_name: str | None, has_aux: bool = False
):
    """Differentiate the global mean loss over replicated parameters.

    Average the scalar loss inside the differentiated function. Autodiff then
    produces the mean parameter gradients; no further gradient reduction is
    applied. Local batches must have equal sizes for this device mean to match
    the mean over all samples.

    Args:
        loss_fn: Function taking parameters first and returning a scalar loss,
            or `(loss, aux)` when `has_aux=True`.
        axis_name: Manual axis in `shard_map` over which losses are averaged. The
            differentiated parameters must be replicated over this axis.
            Pass `None` to use ordinary local `jax.value_and_grad`.
        has_aux: Whether `loss_fn` also returns auxiliary data. Aux is returned
            unchanged on each device and is not differentiated or reduced.

    Returns:
        A callable with the same arguments as `loss_fn`, returning
        `(mean_loss, mean_grads)` or `((mean_loss, local_aux), mean_grads)`.
        With no axis, both the loss and gradients remain local.
    """
    if axis_name is None:
        return jax.value_and_grad(loss_fn, has_aux=has_aux)

    def mean_loss_fn(*args, **kwargs):
        value = loss_fn(*args, **kwargs)
        if has_aux:
            loss, aux = value
            return jax.lax.pmean(loss, axis_name), aux
        return jax.lax.pmean(value, axis_name)

    return jax.value_and_grad(mean_loss_fn, has_aux=has_aux)


def gradient_update(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    axis_name: str | None,
    has_aux: bool = False,
):
    """Wrapper of the loss function that apply gradient updates.

    Args:
        loss_fn: The loss function. (params, ...) -> loss
        optimizer: The optimizer to apply gradients.
        axis_name: Axis for the global mean loss and gradients,
            or `None` for a local update.
        has_aux: Whether the loss_fn has auxiliary data.

    Returns:
        A function that takes the same argument as the loss function plus the
        optimizer state. The output of this function is the loss, the new parameter,
        and the new optimizer state.
    """
    loss_and_grad_fn = pmean_loss_and_grad(
        loss_fn, axis_name=axis_name, has_aux=has_aux
    )

    def f(opt_state, params, *args, **kwargs):
        value, grads = loss_and_grad_fn(params, *args, **kwargs)
        params_update, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, params_update)
        return (
            value,
            params,
            opt_state,
        )

    return f


def pmean_nnx_loss_and_grad(
    loss_fn: Callable,
    axis_name: str | None,
    has_aux: bool = False,
    *,
    wrt=nnx.Param,
):
    """Differentiate selected NNX parameters through the global mean loss.

    `wrt` filters variables in the first model argument. Auxiliary outputs stay
    local. As with `pmean_loss_and_grad`, replicated parameters and equally sized
    local batches are required for data-parallel training.

    Args:
        loss_fn: Function taking an NNX model first and returning a scalar loss,
            or `(loss, aux)` when `has_aux=True`.
        axis_name: Manual `shard_map` axis over which losses are averaged.
            Pass `None` for a local update.
        has_aux: Whether `loss_fn` also returns local auxiliary data.
        wrt: NNX filter selecting the variables to differentiate.

    Returns:
        A callable returning the mean loss and selected gradients, with local
        auxiliary data when requested.
    """

    def mean_loss(*args, **kwargs):
        result = loss_fn(*args, **kwargs)
        if axis_name is None:
            return result
        if has_aux:
            loss, aux = result
            return jax.lax.pmean(loss, axis_name), aux
        return jax.lax.pmean(result, axis_name)

    return nnx.value_and_grad(mean_loss, argnums=nnx.DiffState(0, wrt), has_aux=has_aux)
