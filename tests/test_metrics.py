"""Array defaults follow the instance's execution mesh, independent of imports."""

import dataclasses
from contextlib import nullcontext
import importlib
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.types import PyTreeDict, PyTreeNode, PyTreeData, pytree_field
from evorl.metrics import MetricBase, metric_field


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("evorl.metrics", "WorkflowMetric"),
        ("evorl.metrics", "TrainMetric"),
        ("evorl.metrics", "ECWorkflowMetric"),
        ("evorl.metrics", "MultiObjectiveECWorkflowMetric"),
        ("evorl.algorithms.dqn", "DQNWorkflowMetric"),
        ("evorl.algorithms.dqn", "DQNTrainMetric"),
        ("evorl.algorithms.meta.pbt_workflow", "PBTWorkflowMetric"),
        ("evorl.algorithms.contrib.pop_td3", "WorkflowMetric"),
        ("evorl.algorithms.erl.erl_td3.erl_workflow", "WorkflowMetric"),
        ("evorl.algorithms.contrib.cemrl_origin", "WorkflowMetric"),
        ("evorl.algorithms.contrib.cemrl_origin", "CEMRLTrainMetric"),
        ("evorl.algorithms.contrib.erl_origin", "ERLTrainMetric"),
        ("evorl.algorithms.multi_objective.common", "MultiObjectiveWorkflowMetric"),
        ("evorl.replay_buffers.replay_buffer", "ReplayBufferState"),
        (
            "evorl.replay_buffers.prioritized_replay_buffer",
            "PrioritizedReplayBufferState",
        ),
        ("evorl.workflows.ec_workflow", "DistributedInfo"),
    ],
)
def test_array_defaults_follow_instance_mesh(module_name, class_name, mesh_factory):
    cls = getattr(importlib.import_module(module_name), class_name)

    def create():
        required = {
            field.name: PyTreeDict() if field.name == "data" else jnp.zeros(())
            for field in dataclasses.fields(cls)
            if field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        }
        return cls(**required)

    single = create()
    mesh = mesh_factory()
    with jax.set_mesh(mesh):
        multiple = create()
    again = jax.jit(create)()

    defaults = [
        field.name
        for field in dataclasses.fields(cls)
        if field.default_factory is not dataclasses.MISSING
        and isinstance(getattr(single, field.name), jax.Array)
    ]
    assert defaults
    for name in defaults:
        local, sharded, restored = (
            getattr(value, name) for value in (single, multiple, again)
        )
        assert len(local.sharding.device_set) == 1
        assert sharded.sharding.device_set == set(mesh.devices.flat)
        assert sharded.sharding.is_fully_replicated
        assert len(restored.sharding.device_set) == 1
        np.testing.assert_array_equal(local, sharded)
        np.testing.assert_array_equal(local, restored)
        assert local.dtype == sharded.dtype == restored.dtype


def test_first_import_inside_mesh_does_not_bind_metric_defaults():
    # A fresh process prevents earlier test collection from hiding import order.
    code = textwrap.dedent(
        """
        import jax

        mesh = jax.make_mesh(
            (2,), ('DP',), axis_types=(jax.sharding.AxisType.Explicit,)
        )
        with jax.set_mesh(mesh):
            from evorl.metrics import WorkflowMetric

        metrics = jax.jit(WorkflowMetric)()
        for value in jax.tree.leaves(metrics):
            assert len(value.sharding.device_set) == 1
            assert int(value) == 0
        """
    )
    env = os.environ.copy()
    env.update(JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="2", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("base", "field_fn"),
    [
        (PyTreeNode, None),
        (PyTreeData, None),
        (PyTreeData, pytree_field),
        (MetricBase, metric_field),
    ],
)
@pytest.mark.parametrize(
    "array_kind", ["scalar", "array", "weak_scalar", "weak_array", "key"]
)
@pytest.mark.parametrize("define_in_mesh", [False, True])
def test_captured_array_defaults_rebuild_in_instance_context(
    base, field_fn, array_kind, define_in_mesh, mesh_factory
):
    """Automatic factories detach import-time placement and retain field options."""
    mesh = mesh_factory()
    with jax.set_mesh(mesh) if define_in_mesh else nullcontext():
        if array_kind == "scalar":
            default = jnp.zeros((), dtype=jnp.uint32)
        elif array_kind == "array":
            default = jnp.array([[1.5, -2.0], [3.0, 4.0]], dtype=jnp.float32)
        elif array_kind == "weak_scalar":
            default = jnp.array(2.0)
        elif array_kind == "weak_array":
            default = jnp.stack([jnp.array(1.0), jnp.array(2.0)])
        else:
            default = jax.random.key(13)

        class CapturedDefault(base):
            value: jax.Array = (
                default
                if field_fn is None
                else field_fn(default=default, repr=False, metadata={"marker": "kept"})
            )

    field = dataclasses.fields(CapturedDefault)[0]
    assert field.default is dataclasses.MISSING
    assert callable(field.default_factory)
    if field_fn is not None:
        assert not field.repr
        assert field.metadata["marker"] == "kept"

    single = CapturedDefault().value
    restored = jax.jit(CapturedDefault)().value
    with jax.set_mesh(mesh):
        replicated = CapturedDefault().value
        mapped = jax.jit(
            jax.shard_map(CapturedDefault, in_specs=(), out_specs=jax.P())
        )().value

    assert len(single.sharding.device_set) == 1
    assert len(restored.sharding.device_set) == 1
    assert replicated.sharding.device_set == set(mesh.devices.flat)
    assert mapped.sharding.device_set == set(mesh.devices.flat)
    for value in (single, restored, replicated, mapped):
        assert value.dtype == default.dtype
        assert value.shape == default.shape
        if array_kind == "key":
            np.testing.assert_array_equal(
                jax.random.key_data(value), jax.random.key_data(default)
            )
        else:
            np.testing.assert_array_equal(value, default)
            assert value.weak_type == default.weak_type


@pytest.mark.parametrize("base", [PyTreeNode, PyTreeData, MetricBase])
def test_explicit_factory_survives_class_definition_inside_mesh(base, mesh_factory):
    """A class defined under a mesh still initializes correctly outside it."""
    with jax.set_mesh(mesh_factory()):

        class FreshDefault(base):
            value: jax.Array = pytree_field(
                default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
            )

    value = jax.jit(FreshDefault)().value
    assert len(value.sharding.device_set) == 1
    assert int(value) == 0
