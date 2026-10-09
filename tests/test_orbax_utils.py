"""Round-trip checks for the Orbax V0 checkpoint wrappers."""

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from evorl.utils.orbax_utils import CheckpointManager, load, save, to_shape_dtype_struct


@pytest.fixture(autouse=True)
def mesh_context():
    mesh = jax.make_mesh(
        (1,),
        ("devices",),
        axis_types=(jax.sharding.AxisType.Explicit,),
        devices=jax.devices()[:1],
    )
    with jax.set_mesh(mesh):
        yield


@pytest.mark.parametrize("use_manager", [False, True], ids=["standalone", "manager"])
@pytest.mark.parametrize("target_kind", ["jax", "numpy", "abstract"])
def test_checkpoint_restores_empty_leaves(tmp_path, use_manager, target_kind):
    state = {
        "weights": jax.device_put(jnp.arange(6, dtype=jnp.float32), jax.P()),
        "empty": jax.device_put(jnp.zeros((0, 2), dtype=jnp.float32), jax.P()),
        "nested": (jnp.zeros((3, 0), dtype=jnp.int32), None),
    }
    if target_kind == "numpy":
        state = jax.tree.map(np.asarray, state)
    if target_kind == "abstract":
        target = jax.tree.map(to_shape_dtype_struct, state)
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
        assert restored["empty"].sharding.is_equivalent_to(
            state["empty"].sharding, ndim=2
        )


@pytest.mark.parametrize("use_manager", [False, True], ids=["standalone", "manager"])
@pytest.mark.parametrize("abstract", [False, True])
@pytest.mark.parametrize("impl", ["threefry2x32", "rbg"])
def test_checkpoint_typed_keys(tmp_path, use_manager, abstract, impl):
    key = jax.random.key(41, impl=impl)
    state = {
        "key": key,
        "keys": jax.random.split(key, (2, 3)),
        "empty": jax.device_put(jax.random.split(key, (0, 2)), jax.P()),
        "data": jnp.arange(4, dtype=jnp.uint32),
    }
    target = jax.tree.map(to_shape_dtype_struct, state) if abstract else state
    if use_manager:
        with CheckpointManager(tmp_path / "managed") as manager:
            assert manager.save(0, state)
            manager.wait_until_finished()
            restored = manager.restore(0, target)
    else:
        save(tmp_path / "checkpoint", state)
        restored = load(tmp_path / "checkpoint", target)

    for name in ("key", "keys", "empty"):
        assert restored[name].shape == state[name].shape
        assert restored[name].dtype == state[name].dtype
        np.testing.assert_array_equal(
            jax.random.key_data(restored[name]), jax.random.key_data(state[name])
        )
    assert restored["empty"].sharding.is_equivalent_to(state["empty"].sharding, ndim=2)
    np.testing.assert_array_equal(restored["data"], state["data"])
    np.testing.assert_array_equal(
        jax.random.normal(restored["key"], (5,)),
        jax.random.normal(state["key"], (5,)),
    )


def test_disabled_manager_v0_interface_and_workflow_cleanup(tmp_path):
    from unittest.mock import Mock
    from omegaconf import OmegaConf
    from evorl.utils.orbax_utils import DummyCheckpointManager, setup_checkpoint_manager
    from evorl.workflows.workflow import Workflow

    manager = DummyCheckpointManager(tmp_path / "disabled")
    assert manager.directory == tmp_path / "disabled"
    assert isinstance(manager.metadata(), ocp.metadata.RootMetadata)
    assert isinstance(manager.metadata(0), ocp.metadata.StepMetadata)
    assert not manager.save(0, {})
    assert not manager.should_save(0)
    assert not manager.reached_preemption(0)
    assert not manager.directory.exists()
    disabled = setup_checkpoint_manager(
        OmegaConf.create({"checkpoint": {"enable": False}})
    )
    assert isinstance(disabled, DummyCheckpointManager)
    workflow = Mock()
    workflow.recorder.close.side_effect = RuntimeError("recorder failed")
    with pytest.raises(RuntimeError, match="recorder failed"):
        Workflow.close(workflow)
    workflow.checkpoint_manager.close.assert_called_once()
