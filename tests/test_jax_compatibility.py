"""Regression checks shared by the minimum and latest JAX environments."""

import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import optax

from evorl.algorithms.td7 import avg_l1_norm
from evorl.distributed import get_process_id, is_dist_initialized
from evorl.networks.linear import MLP


def test_linen_functional_training_step():
    model = MLP(layer_sizes=(8, 2))
    inputs = jnp.ones((4, 3))
    params = model.init(jax.random.PRNGKey(0), inputs)
    optimizer = optax.adam(1e-2)

    def loss(params):
        return jnp.mean(model.apply(params, inputs) ** 2)

    @jax.jit
    def step(params, opt_state):
        grads = jax.grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    updated, _ = step(params, optimizer.init(params))
    assert loss(updated) < loss(params)


def test_td7_normalization_handles_zero_and_nonzero_inputs():
    inputs = jnp.array([[0.0, 0.0], [1.0, -3.0]])
    normalized = jax.jit(avg_l1_norm)(inputs)
    np.testing.assert_allclose(normalized, [[0.0, 0.0], [0.5, -1.5]])
    grads = jax.grad(lambda x: jnp.sum(avg_l1_norm(x)))(inputs)
    assert jnp.all(jnp.isfinite(grads))


def test_single_process_distributed_status():
    assert not is_dist_initialized()
    assert get_process_id() == 0


def test_shard_map_wrappers_and_gradients_on_two_cpu_devices():
    code = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from evorl.distributed import shmap_map, shmap_vmap

        assert jax.device_count() == 2
        mesh = jax.sharding.Mesh(jax.devices(), ("devices",))
        P = jax.sharding.PartitionSpec
        inputs = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)

        for wrapper in (shmap_map, shmap_vmap):
            mapped = jax.jit(wrapper(
                lambda row: row ** 2,
                mesh=mesh, in_specs=P("devices"), out_specs=P("devices"),
                check_vma=False,
            ))
            np.testing.assert_allclose(mapped(inputs), inputs ** 2)
            grads = jax.grad(lambda x: jnp.sum(mapped(x)))(inputs)
            np.testing.assert_allclose(grads, 2 * inputs)

        reduced = jax.jit(jax.shard_map(
            lambda block: jax.lax.pmean(block.sum(axis=0), "devices"),
            mesh=mesh, in_specs=P("devices"), out_specs=P(),
            check_vma=False,
        ))
        np.testing.assert_allclose(reduced(inputs), inputs.sum(axis=0) / 2)
        grads = jax.grad(lambda x: jnp.sum(reduced(x)))(inputs)
        np.testing.assert_allclose(grads, jnp.full_like(inputs, 0.5))
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            "JAX_NUM_CPU_DEVICES": "2",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
