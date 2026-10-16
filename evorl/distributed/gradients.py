from collections.abc import Callable

import chex
import jax
import optax

"""Training gradient utility functions.

Modified from https://github.com/google/brax/blob/main/brax/training/gradients.py
"""


def pmean_loss_and_grad(
    loss_fn: Callable, dp_axis_name: str | None, has_aux: bool = False
):
    """Differentiate the global mean loss over replicated parameters.

    Average the scalar loss inside the differentiated function. Autodiff then
    produces the mean parameter gradients; no further gradient reduction is
    applied. Local batches must have equal sizes for this device mean to match
    the mean over all samples.

    Args:
        loss_fn: Function taking parameters first and returning a scalar loss,
            or `(loss, aux)` when `has_aux=True`.
        dp_axis_name: Active data-parallel Manual axis in `shard_map`. The
            differentiated parameters must be replicated over this axis.
            Pass `None` to use ordinary local `jax.value_and_grad`.
        has_aux: Whether `loss_fn` also returns auxiliary data. Aux is returned
            unchanged on each device and is not differentiated or reduced.

    Returns:
        A callable with the same arguments as `loss_fn`, returning
        `(mean_loss, mean_grads)` or `((mean_loss, local_aux), mean_grads)`.
        With no axis, both the loss and gradients remain local.
    """
    if dp_axis_name is None:
        return jax.value_and_grad(loss_fn, has_aux=has_aux)

    def mean_loss_fn(*args, **kwargs):
        value = loss_fn(*args, **kwargs)
        if has_aux:
            loss, aux = value
            return jax.lax.pmean(loss, dp_axis_name), aux
        return jax.lax.pmean(value, dp_axis_name)

    return jax.value_and_grad(mean_loss_fn, has_aux=has_aux)


def gradient_update(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    dp_axis_name: str | None,
    has_aux: bool = False,
):
    """Wrapper of the loss function that apply gradient updates.

    Args:
        loss_fn: The loss function. (params, ...) -> loss
        optimizer: The optimizer to apply gradients.
        dp_axis_name: Data-parallel axis for the global mean loss and gradients,
            or `None` for a local update.
        has_aux: Whether the loss_fn has auxiliary data.

    Returns:
        A function that takes the same argument as the loss function plus the
        optimizer state. The output of this function is the loss, the new parameter,
        and the new optimizer state.
    """
    loss_and_grad_fn = pmean_loss_and_grad(
        loss_fn, dp_axis_name=dp_axis_name, has_aux=has_aux
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


def _attach_params_to_agent_state(agent_state, params):
    return agent_state.replace(params=params)


def _detach_params_to_agent_state(agent_state):
    return agent_state.params


def agent_gradient_update(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    dp_axis_name: str | None = None,
    has_aux: bool = False,
    attach_fn: Callable[
        [chex.ArrayTree, chex.ArrayTree], chex.ArrayTree
    ] = _attach_params_to_agent_state,
    detach_fn: Callable[
        [chex.ArrayTree, chex.ArrayTree], chex.ArrayTree
    ] = _detach_params_to_agent_state,
):
    def _loss_fn(params, agent_state, sample_batch, key):
        agent_state = attach_fn(agent_state, params)
        return loss_fn(agent_state, sample_batch, key)

    _gradient_update_fn = gradient_update(
        _loss_fn, optimizer, dp_axis_name=dp_axis_name, has_aux=has_aux
    )

    def f(opt_state, agent_state, *args, **kwargs):
        params = detach_fn(agent_state)
        value, params, opt_state = _gradient_update_fn(
            opt_state, params, agent_state, *args, **kwargs
        )
        agent_state = attach_fn(agent_state, params)

        return value, agent_state, opt_state

    return f
