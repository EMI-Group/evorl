"""Exercise Explicit mesh boundaries and global population operations."""

import os
from pathlib import Path
import subprocess
import sys


def test_explicit_workflows_on_two_cpu_devices():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("_explicit_sharding_cases.py")),
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
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
