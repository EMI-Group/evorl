"""Run the workflow boundaries on two CPU devices."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_pbt_targets_on_independent_meta_population_axis():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("_pbt_workflow_multidevice_cases.py")),
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
    assert result.returncode == 0, result.stdout + result.stderr


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
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize(
    "agent",
    [
        "ppo",
        "sac",
        "dqn",
        "td3",
        "ddpg",
        "contrib/td3-v3",
        "td7",
        "ec/openes",
        "random",
        "a2c",
        "impala",
        "contrib/td3-v2",
        "contrib/td3-onpolicy",
    ],
)
def test_workflow_methods_on_two_cpu_devices(jit, agent):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("_workflow_multidevice_cases.py")),
            str(int(jit)),
            agent,
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
    assert result.returncode == 0, result.stdout + result.stderr
