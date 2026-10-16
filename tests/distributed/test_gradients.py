"""Global mean gradients, local auxiliaries and optimizer updates."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from evorl.agent import AgentState
from evorl.distributed.gradients import (
    agent_gradient_update,
    gradient_update,
    pmean_loss_and_grad,
)
from evorl.metrics import TrainMetric


def _data_loss(w, batch):
    return ((w - batch) ** 2).mean()


@pytest.mark.parametrize("has_aux", [False, True], ids=["loss_only", "with_aux"])
def test_no_axis_keeps_local_loss_and_aux(has_aux, jit):
    def loss_fn(w, targets):
        errors = (w - targets) ** 2
        return (errors.mean(), errors) if has_aux else errors.mean()

    fn = pmean_loss_and_grad(loss_fn, None, has_aux=has_aux)
    if jit:
        fn = jax.jit(fn)
    value, grad = fn(jnp.float32(0), jnp.array([1.0, 3.0]))
    if has_aux:
        value, aux = value
        np.testing.assert_array_equal(aux, [1.0, 9.0])
    np.testing.assert_allclose(value, 5.0)
    np.testing.assert_allclose(grad, -4.0)


@pytest.mark.parametrize(
    "kind,expected_loss,expected_grad",
    [("data", 5.0, -4.0), ("regularizer", 1.0, -2.0), ("mixed", 6.0, -6.0)],
    ids=["data", "parameter_regularizer", "mixed"],
)
@pytest.mark.parametrize("has_aux", [False, True], ids=["loss_only", "with_aux"])
def test_global_mean_gradients_and_local_aux(
    dp_mesh, jit, kind, expected_loss, expected_grad, has_aux
):
    targets = jax.device_put(jnp.array([1.0, 3.0]), jax.P("DP"))
    w = jax.device_put(jnp.float32(0), jax.P())

    def loss_fn(w, batch):
        errors = (w - batch) ** 2
        regularizer = (w - 1.0) ** 2
        loss = errors.mean() if kind == "data" else regularizer
        if kind == "mixed":
            loss = errors.mean() + regularizer
        return (loss, errors) if has_aux else loss

    fn = jax.shard_map(
        pmean_loss_and_grad(loss_fn, "DP", has_aux=has_aux),
        in_specs=(jax.P(), jax.P("DP")),
        out_specs=((jax.P(), jax.P("DP")), jax.P()) if has_aux else (jax.P(), jax.P()),
    )
    if jit:
        fn = jax.jit(fn)
    value, grad = fn(w, targets)
    if has_aux:
        value, priorities = value
        np.testing.assert_array_equal(priorities, [1.0, 9.0])
        for rank, shard in enumerate(priorities.addressable_shards):
            np.testing.assert_array_equal(shard.data, [[1.0], [9.0]][rank])
    np.testing.assert_allclose(value, expected_loss)
    np.testing.assert_allclose(grad, expected_grad)
    for shard in grad.addressable_shards:
        np.testing.assert_allclose(shard.data, expected_grad)


def test_reporting_global_loss_adds_no_reduction(dp_mesh):
    base = pmean_loss_and_grad(_data_loss, "DP")

    def report(w, batch):
        value, grad = base(w, batch)
        metric = TrainMetric(loss=value).all_reduce("DP")
        return metric.loss, grad

    w = jax.device_put(jnp.float32(0), jax.P())
    targets = jax.device_put(jnp.array([1.0, 3.0]), jax.P("DP"))
    counts = []
    for fn in (base, report):
        mapped = jax.jit(
            jax.shard_map(
                fn, in_specs=(jax.P(), jax.P("DP")), out_specs=(jax.P(), jax.P())
            )
        )
        compiled = mapped.lower(w, targets).compile()
        counts.append(
            sum(" all-reduce(" in line for line in compiled.as_text().splitlines())
        )
        value, grad = compiled(w, targets)
        np.testing.assert_allclose(value, 5.0)
        np.testing.assert_allclose(grad, -4.0)
    assert counts[0] > 0
    assert counts[0] == counts[1], counts


@pytest.mark.parametrize("use_agent_state", [False, True], ids=["params", "agent"])
def test_optimizer_update_preserves_local_priorities(dp_mesh, jit, use_agent_state):
    params = {"w": jax.device_put(jnp.float32(0), jax.P())}
    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(params)
    agent = AgentState(params=params)
    targets = jax.device_put(jnp.array([1.0, 3.0]), jax.P("DP"))

    def loss_fn(params, batch):
        errors = (params["w"] - batch) ** 2
        return errors.mean(), errors

    def update(params, opt_state, batch):
        return gradient_update(loss_fn, optimizer, "DP", has_aux=True)(
            opt_state, params, batch
        )

    def update_agent(agent, opt_state, batch):
        def agent_loss(state, batch, key):
            return loss_fn(state.params, batch)

        return agent_gradient_update(agent_loss, optimizer, "DP", has_aux=True)(
            opt_state, agent, batch, jax.random.key(0)
        )

    initial = agent if use_agent_state else params
    fn = jax.shard_map(
        update_agent if use_agent_state else update,
        in_specs=(jax.P(), jax.P(), jax.P("DP")),
        out_specs=((jax.P(), jax.P("DP")), jax.P(), jax.P()),
    )
    if jit:
        fn = jax.jit(fn)
    (value, aux), updated, _ = fn(initial, opt_state, targets)
    updated = updated.params if use_agent_state else updated
    np.testing.assert_allclose(value, 5.0)
    np.testing.assert_array_equal(aux, [1.0, 9.0])
    np.testing.assert_allclose(updated["w"], 0.4)
