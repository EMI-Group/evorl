"""Check mean-loss gradients and unchanged device-local auxiliary data."""

import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evorl.distributed.gradients import pmean_loss_and_grad


@pytest.mark.parametrize("has_aux", [False, True])
def test_no_axis_keeps_local_loss_and_aux(has_aux):
    def loss_fn(w, targets):
        errors = (w - targets) ** 2
        return (errors.mean(), errors) if has_aux else errors.mean()

    fn = jax.jit(pmean_loss_and_grad(loss_fn, None, has_aux=has_aux))
    value, grad = fn(jnp.float32(0), jnp.array([1.0, 3.0]))
    if has_aux:
        value, aux = value
        np.testing.assert_array_equal(aux, [1.0, 9.0])
    np.testing.assert_allclose(value, 5.0)
    np.testing.assert_allclose(grad, -4.0)


@pytest.mark.parametrize("jit", [False, True])
def test_global_mean_gradients_and_local_aux_on_two_devices(jit):
    code = textwrap.dedent(
        """
        import sys
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from evorl.agent import AgentState
        from evorl.distributed.gradients import (
            pmean_loss_and_grad, gradient_update, agent_gradient_update,
        )
        from evorl.metrics import TrainMetric

        jit = bool(int(sys.argv[1]))
        mesh = jax.make_mesh((2,), ("DP",))
        with jax.set_mesh(mesh):
            targets = jax.device_put(jnp.array([1., 3.]), jax.P("DP"))
            w = jax.device_put(jnp.float32(0), jax.P())
            for kind, expected_loss, expected_grad in (
                ("data", 5., -4.),
                ("regularizer", 1., -2.),
                ("mixed", 6., -6.),
            ):
                for has_aux in (False, True):
                    def loss_fn(w, batch):
                        errors = (w - batch) ** 2
                        regularizer = (w - 1.) ** 2
                        loss = errors.mean() if kind == "data" else regularizer
                        if kind == "mixed":
                            loss = errors.mean() + regularizer
                        return (loss, errors) if has_aux else loss

                    fn = jax.shard_map(
                        pmean_loss_and_grad(loss_fn, "DP", has_aux=has_aux),
                        in_specs=(jax.P(), jax.P("DP")),
                        out_specs=((jax.P(), jax.P("DP")), jax.P())
                            if has_aux else (jax.P(), jax.P()),
                    )
                    if jit:
                        fn = jax.jit(fn)
                    value, grad = fn(w, targets)
                    if has_aux:
                        value, priorities = value
                        np.testing.assert_array_equal(priorities, [1., 9.])
                        for rank, shard in enumerate(priorities.addressable_shards):
                            np.testing.assert_array_equal(shard.data, [[1.], [9.]][rank])
                    np.testing.assert_allclose(value, expected_loss)
                    np.testing.assert_allclose(grad, expected_grad)
                    for shard in grad.addressable_shards:
                        np.testing.assert_allclose(shard.data, expected_grad)

            # Reporting an already-global primary loss must not add a second
            # loss reduction to the differentiated computation.
            base = pmean_loss_and_grad(
                lambda w, batch: ((w - batch) ** 2).mean(), "DP",
            )
            def report(w, batch):
                value, grad = base(w, batch)
                metric = TrainMetric(loss=value).all_reduce("DP")
                return metric.loss, grad

            reduction_counts = []
            for fn in (base, report):
                mapped = jax.jit(jax.shard_map(
                    fn, in_specs=(jax.P(), jax.P("DP")),
                    out_specs=(jax.P(), jax.P()),
                ))
                compiled = mapped.lower(w, targets).compile()
                reduction_counts.append(sum(
                    " all-reduce(" in line
                    for line in compiled.as_text().splitlines()
                ))
                value, grad = compiled(w, targets)
                np.testing.assert_allclose(value, 5.)
                np.testing.assert_allclose(grad, -4.)
            assert reduction_counts[0] > 0
            assert reduction_counts[0] == reduction_counts[1], reduction_counts

            # Exercise both optimizer wrappers with a parameter pytree and
            # per-sample priorities that must not be averaged between devices.
            params = {"w": w}
            optimizer = optax.sgd(0.1)
            opt_state = optimizer.init(params)
            agent = AgentState(params=params)
            def loss_fn(params, batch):
                errors = (params["w"] - batch) ** 2
                return errors.mean(), errors

            def update(params, opt_state, batch):
                return gradient_update(
                    loss_fn, optimizer, "DP", has_aux=True,
                )(opt_state, params, batch)

            def update_agent(agent, opt_state, batch):
                def agent_loss(state, batch, key):
                    return loss_fn(state.params, batch)
                return agent_gradient_update(
                    agent_loss, optimizer, "DP", has_aux=True,
                )(opt_state, agent, batch, jax.random.key(0))

            for fn, initial in ((update, params), (update_agent, agent)):
                fn = jax.shard_map(
                    fn,
                    in_specs=(jax.P(), jax.P(), jax.P("DP")),
                    out_specs=((jax.P(), jax.P("DP")), jax.P(), jax.P()),
                )
                if jit:
                    fn = jax.jit(fn)
                (value, aux), updated, _ = fn(initial, opt_state, targets)
                updated = updated.params if isinstance(updated, AgentState) else updated
                np.testing.assert_allclose(value, 5.)
                np.testing.assert_array_equal(aux, [1., 9.])
                np.testing.assert_allclose(updated["w"], 0.4)
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(int(jit))],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_NUM_CPU_DEVICES": "2",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        },
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
