"""Array defaults follow the instance's execution mesh, independent of imports."""

import dataclasses
import importlib
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.types import PyTreeDict


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
