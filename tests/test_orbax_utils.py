"""Round-trip checks for the Orbax V0 checkpoint wrappers."""

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from evorl.utils.orbax_utils import CheckpointManager, load, save


@pytest.mark.parametrize("use_manager", [False, True], ids=["standalone", "manager"])
@pytest.mark.parametrize("target_kind", ["jax", "numpy", "abstract"])
def test_checkpoint_restores_empty_leaves(tmp_path, use_manager, target_kind):
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices()[:1], ("devices",)),
        jax.sharding.PartitionSpec(),
    )
    state = {
        "weights": jax.device_put(jnp.arange(6, dtype=jnp.float32), sharding),
        "empty": jax.device_put(jnp.zeros((0, 2), dtype=jnp.float32), sharding),
        "nested": (jnp.zeros((3, 0), dtype=jnp.int32), None),
    }
    if target_kind == "numpy":
        state = jax.tree.map(np.asarray, state)
    if target_kind == "abstract":
        target = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
    else:
        target = jax.tree.map(lambda x: x * 0, state)

    if use_manager:
        options = ocp.CheckpointManagerOptions(save_interval_steps=2, max_to_keep=1)
        with CheckpointManager(tmp_path / "managed", options=options) as manager:
            assert manager.save(0, state)
            assert not manager.save(1, state)
            manager.wait_until_finished()
            restored = manager.restore(0, target)
            assert manager.latest_step() == 0
    else:
        path = tmp_path / "standalone"
        save(path, state)
        restored = load(path, target)

    assert jax.tree.structure(restored) == jax.tree.structure(state)
    for expected, actual in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
    if target_kind == "abstract":
        assert restored["empty"].sharding.is_equivalent_to(sharding, ndim=2)
