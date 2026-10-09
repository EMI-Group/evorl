"""Run the workflow boundaries on two CPU devices."""

import os
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("jit", [True, False])
def test_independent_replay_rings_on_two_cpu_devices(jit):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("_replay_sharding_cases.py")),
            str(int(jit)),
        ],
        env={
            **os.environ,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            "JAX_NUM_CPU_DEVICES": "2",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    if (
        version("jax") == "0.7.1"
        and not jit
        and "ShardingTypeError: Use `.at[...].get(out_sharding=)`" in result.stderr
    ):
        pytest.xfail(
            "JAX 0.7.1 eager shard_map fails on fold_in(key, axis_index); "
            "compiled initialization and step work."
        )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("jit", [True, False])
def test_workflow_methods_on_two_cpu_devices(jit):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("_workflow_multidevice_cases.py")),
            str(int(jit)),
        ],
        env={
            **os.environ,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            "JAX_NUM_CPU_DEVICES": "2",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    if (
        version("jax") == "0.7.1"
        and not jit
        and "ShardingTypeError: Use `.at[...].get(out_sharding=)`" in result.stderr
    ):
        pytest.xfail(
            "JAX 0.7.1 eager shard_map fails on fold_in(key, axis_index) or "
            "replicated arrays captured by scan; "
            "also reproducible with scan(lambda x, _: (x @ jnp.asarray(w), ()), ...). "
            "Outer jit works, as does eager execution on current JAX."
        )
    assert result.returncode == 0, result.stdout + result.stderr
