"""Manual sharding, collective transposes and scoped VMA casts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evorl.distributed import all_gather, shmap_map, shmap_vmap
from evorl.distributed.sharding import mark_varying

from tests.helpers.sharding import compiled_collectives


@pytest.mark.parametrize("wrapper", [shmap_map, shmap_vmap], ids=["map", "vmap"])
def test_local_maps_preserve_values_and_gradients(dp_mesh, wrapper):
    inputs = jax.device_put(jnp.arange(8, dtype=jnp.float32).reshape(4, 2), jax.P("DP"))
    mapped = jax.jit(wrapper(lambda row: row**2, jax.P("DP"), jax.P("DP")))
    np.testing.assert_allclose(mapped(inputs), inputs**2)
    grads = jax.jit(jax.grad(lambda x: jnp.sum(mapped(x))))(inputs)
    np.testing.assert_allclose(grads, 2 * inputs)
    assert not compiled_collectives(mapped.lower(inputs).compile().as_text())


def test_local_vmap_preserves_individual_random_streams(dp_mesh):
    keys = jax.device_put(jax.random.split(jax.random.key(7), 4), jax.P("DP"))
    assert keys.shape == (4,)
    assert keys.addressable_shards[0].data.shape == (2,)
    sample = jax.jit(shmap_vmap(jax.random.uniform, jax.P("DP"), jax.P("DP")))
    np.testing.assert_array_equal(sample(keys), jax.vmap(jax.random.uniform)(keys))


def test_mean_collective_gradient(dp_mesh):
    inputs = jax.device_put(jnp.arange(8, dtype=jnp.float32).reshape(4, 2), jax.P("DP"))
    reduced = jax.jit(
        jax.shard_map(
            lambda block: jax.lax.pmean(block.sum(axis=0), "DP"),
            in_specs=jax.P("DP"),
            out_specs=jax.P(),
        )
    )
    np.testing.assert_allclose(reduced(inputs), inputs.sum(axis=0) / 2)
    grads = jax.jit(jax.grad(lambda x: jnp.sum(reduced(x))))(inputs)
    np.testing.assert_allclose(grads, jnp.full_like(inputs, 0.5))


def test_invariant_all_gather_and_gradient(dp_mesh):
    inputs = jax.device_put(jnp.arange(8, dtype=jnp.float32).reshape(4, 2), jax.P("DP"))
    gathered = jax.jit(
        jax.shard_map(
            lambda block: all_gather(block, "DP", axis=0, tiled=True, to="invarying"),
            in_specs=jax.P("DP"),
            out_specs=jax.P(),
        )
    )
    np.testing.assert_array_equal(gathered(inputs), inputs)
    grads = jax.jit(jax.grad(lambda x: jnp.sum(gathered(x) ** 2)))(inputs)
    np.testing.assert_allclose(grads, 2 * inputs)
    assert compiled_collectives(gathered.lower(inputs).compile().as_text()) == {
        "all-gather"
    }


@pytest.mark.four_devices
def test_mark_varying_only_changes_supplied_manual_axes(mesh_factory):
    mesh = mesh_factory((2, 2), ("DP", "POP"))
    with jax.set_mesh(mesh):

        @jax.shard_map(in_specs=jax.P("DP"), out_specs=jax.P("DP"))
        def accumulate(block):
            initial = jnp.zeros(block.shape)
            unchanged = mark_varying(initial, None)
            assert not jax.typeof(unchanged).manual_axis_type.varying
            carry = mark_varying({"total": unchanged, "block": block}, ("DP",))
            carry = mark_varying(carry, "DP")

            def step(carry, _):
                return {
                    "total": carry["total"] + carry["block"],
                    "block": carry["block"],
                }, ()

            final, _ = jax.lax.scan(step, carry, (), length=2)
            return final["total"]

        inputs = jax.device_put(jnp.arange(4, dtype=jnp.float32), jax.P("DP"))
        compiled = jax.jit(accumulate).lower(inputs).compile()
        np.testing.assert_array_equal(compiled(inputs), 2 * inputs)
        assert not compiled_collectives(compiled.as_text())


def test_mark_varying_requires_an_explicit_axis():
    local = jnp.arange(2)
    assert mark_varying(local, None) is local
    assert mark_varying(local, ()) is local
    with pytest.raises(TypeError):
        mark_varying(local)
