from collections.abc import Callable

import chex
import jax
import optax

"""Training gradient utility functions.

Modified from https://github.com/google/brax/blob/main/brax/training/gradients.py
"""


def loss_and_pgrad(
    loss_fn: Callable[..., float], pmap_axis_name: str | None, has_aux: bool = False
):
    g = jax.value_and_grad(loss_fn, has_aux=has_aux)

    def h(*args, **kwargs):
        value, grads = g(*args, **kwargs)
        return value, jax.lax.pmean(grads, axis_name=pmap_axis_name)

    return g if pmap_axis_name is None else h


def gradient_update(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: str | None,
    has_aux: bool = False,
):
    """Wrapper of the loss function that apply gradient updates.

    Args:
        loss_fn: The loss function. (params, ...) -> loss
        optimizer: The optimizer to apply gradients.
        pmap_axis_name: If relevant, the name of the pmap axis to synchronize gradients.
        has_aux: Whether the loss_fn has auxiliary data.

    Returns:
        A function that takes the same argument as the loss function plus the
        optimizer state. The output of this function is the loss, the new parameter,
        and the new optimizer state.
    """
    loss_and_pgrad_fn = loss_and_pgrad(
        loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
    )

    def f(opt_state, params, *args, **kwargs):
        value, grads = loss_and_pgrad_fn(params, *args, **kwargs)
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
    pmap_axis_name: str | None = None,
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
        _loss_fn, optimizer, pmap_axis_name=pmap_axis_name, has_aux=has_aux
    )

    def f(opt_state, agent_state, *args, **kwargs):
        params = detach_fn(agent_state)
        value, params, opt_state = _gradient_update_fn(
            opt_state, params, agent_state, *args, **kwargs
        )
        agent_state = attach_fn(agent_state, params)

        return value, agent_state, opt_state

    return f
