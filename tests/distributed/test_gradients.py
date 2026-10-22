"""Global mean gradients, local auxiliaries and optimizer updates."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from evorl.distributed.gradients import (
    pmean_nnx_loss_and_grad,
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
        pmean_loss_and_grad(loss_fn, axis_name="DP", has_aux=has_aux),
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


@pytest.mark.parametrize("use_nnx", [False, True], ids=["params", "nnx"])
def test_optimizer_update_preserves_local_priorities(dp_mesh, jit, use_nnx):
    params = {"w": jax.device_put(jnp.float32(0), jax.P())}
    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(params)
    targets = jax.device_put(jnp.array([1.0, 3.0]), jax.P("DP"))

    def loss_fn(params, batch):
        errors = (params["w"] - batch) ** 2
        return errors.mean(), errors

    def update(params, opt_state, batch):
        return gradient_update(loss_fn, optimizer, "DP", has_aux=True)(
            opt_state, params, batch
        )

    model = nnx.Dict(
        actor=nnx.Dict(w=nnx.Param(params["w"])),
        critic=nnx.Dict(w=nnx.Param(jnp.float32(5))),
    )
    graphdef, model_state = nnx.split(model)
    nnx_opt_state = nnx.unpack(nnx.Optimizer(model.actor, optimizer, wrt=nnx.Param))

    def update_model(model_state, opt_state, batch):
        model = nnx.merge(graphdef, model_state, copy=True)

        def model_loss(model, batch):
            errors = (model.actor.w[...] - batch) ** 2
            return errors.mean(), errors

        value, grads = pmean_nnx_loss_and_grad(
            model_loss,
            "DP",
            has_aux=True,
            wrt=nnx.All(nnx.Param, nnx.PathContains("actor")),
        )(model, batch)
        assert set(grads) == {"actor"}
        local_optimizer = nnx.merge(opt_state, copy=True)
        local_optimizer.update(model.actor, grads["actor"])
        return value, nnx.state(model), nnx.unpack(local_optimizer)

    initial = model_state if use_nnx else params
    opt_state = nnx_opt_state if use_nnx else opt_state
    fn = jax.shard_map(
        update_model if use_nnx else update,
        in_specs=(jax.P(), jax.P(), jax.P("DP")),
        out_specs=((jax.P(), jax.P("DP")), jax.P(), jax.P()),
    )
    if jit:
        fn = jax.jit(fn)
    (value, aux), updated, _ = fn(initial, opt_state, targets)
    if use_nnx:
        np.testing.assert_allclose(updated["critic"]["w"][...], 5)
        updated = updated["actor"]
        np.testing.assert_allclose(model_state["actor"]["w"][...], 0)
    np.testing.assert_allclose(value, 5.0)
    np.testing.assert_array_equal(aux, [1.0, 9.0])
    actual = updated["w"][...] if use_nnx else updated["w"]
    np.testing.assert_allclose(actual, 0.4)


@pytest.mark.parametrize(
    "kind,expected_loss,expected_grad",
    [
        ("data", 5.0, -4.0),
        ("regularizer", 1.0, -2.0),
        ("mixed", 6.0, -6.0),
    ],
)
@pytest.mark.parametrize("has_aux", [False, True])
def test_nnx_selected_global_mean_gradient_and_local_aux(
    dp_mesh, jit, kind, expected_loss, expected_grad, has_aux
):
    model = nnx.Dict(
        actor=nnx.Dict(w=nnx.Param(jnp.float32(0))),
        critic=nnx.Dict(w=nnx.Param(jnp.float32(5))),
    )
    graphdef, state = nnx.split(model)
    targets = jax.device_put(jnp.array([1.0, 3.0]), jax.P("DP"))

    def loss(model, batch):
        w = model.actor.w[...]
        errors = (w - batch) ** 2
        regularizer = (w - 1.0) ** 2
        loss = errors.mean() if kind == "data" else regularizer
        if kind == "mixed":
            loss = errors.mean() + regularizer
        return (loss, errors) if has_aux else loss

    def differentiate(state, batch):
        model = nnx.merge(graphdef, state, copy=True)
        return pmean_nnx_loss_and_grad(
            loss,
            axis_name="DP",
            has_aux=has_aux,
            wrt=nnx.All(nnx.Param, nnx.PathContains("actor")),
        )(model, batch)

    fn = jax.shard_map(
        differentiate,
        in_specs=(jax.P(), jax.P("DP")),
        out_specs=((jax.P(), jax.P("DP")), jax.P()) if has_aux else (jax.P(), jax.P()),
    )
    if jit:
        fn = jax.jit(fn)
    value, grads = fn(state, targets)
    if has_aux:
        value, aux = value
        np.testing.assert_array_equal(aux, [1.0, 9.0])
    np.testing.assert_allclose(value, expected_loss)
    assert set(grads) == {"actor"}
    np.testing.assert_allclose(grads["actor"]["w"][...], expected_grad)


@pytest.mark.parametrize("use_nnx", [False, True])
def test_mean_gradient_accepts_an_arbitrary_axis_name(mesh_factory, use_nnx, jit):
    mesh = mesh_factory(names=("WORKERS",))
    model = nnx.Dict(w=nnx.Param(jnp.float32(0)))
    snapshot = nnx.unpack(model)

    def local_loss(model, targets):
        return _data_loss(model.w[...], targets)

    def differentiate(snapshot, targets):
        model = nnx.merge(snapshot, copy=True)
        loss, grads = pmean_nnx_loss_and_grad(local_loss, axis_name="WORKERS")(
            model, targets
        )
        return loss, grads["w"][...]

    fn = (
        differentiate
        if use_nnx
        else pmean_loss_and_grad(_data_loss, axis_name="WORKERS")
    )
    with jax.set_mesh(mesh):
        fn = jax.shard_map(
            fn, in_specs=(jax.P(), jax.P("WORKERS")), out_specs=(jax.P(), jax.P())
        )
        if jit:
            fn = jax.jit(fn)
        inputs = snapshot if use_nnx else jnp.float32(0)
        targets = jax.device_put(jnp.array([1.0, 3.0]), jax.P("WORKERS"))
        loss, grad = fn(inputs, targets)
    np.testing.assert_allclose(loss, 5.0)
    np.testing.assert_allclose(grad, -4.0)
