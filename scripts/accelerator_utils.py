"""Discover JAX GPUs in a subprocess without initializing the caller's backend."""

import json
import os
import subprocess
import sys
from pathlib import Path

GPU_VISIBLE_DEVICE_ENV = {
    "cuda": "JAX_CUDA_VISIBLE_DEVICES",
    "rocm": "JAX_ROCM_VISIBLE_DEVICES",
}


def probe_gpus() -> dict:
    """Return the default GPU backend and its runtime-visible hardware IDs."""
    env = os.environ.copy()
    # The probe needs a client, but no device arrays or training allocator pool.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--probe"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("JAX GPU discovery timed out after 60 seconds") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"JAX GPU discovery failed:\n{error.stderr.strip()}"
        ) from error
    return json.loads(result.stdout)


def gpu_visibility_settings(job_id: int) -> tuple[dict[str, str], dict, int]:
    """Probe GPUs and return one job's settings, device, and visible GPU count."""
    probe = probe_gpus()
    settings, device = _gpu_visibility_settings(probe, job_id)
    return settings, device, len(probe["devices"])


def _gpu_visibility_settings(probe: dict, job_id: int) -> tuple[dict[str, str], dict]:
    """Return JAX settings selecting one GPU within existing runtime masks."""
    platform = probe["platform"]
    if platform not in GPU_VISIBLE_DEVICE_ENV:
        raise ValueError(f"Unsupported GPU backend: {platform}")
    devices = probe["devices"]
    if not devices:
        raise ValueError("No visible CUDA or ROCm GPUs")
    device = devices[job_id % len(devices)]
    settings = {
        "JAX_PLATFORMS": platform,
        GPU_VISIBLE_DEVICE_ENV[platform]: str(device["local_hardware_id"]),
    }
    return settings, device


def _probe() -> dict:
    # Only this short-lived process imports JAX. Plugin output belongs on stderr.
    import contextlib

    with contextlib.redirect_stdout(sys.stderr):
        from jax.extend import backend

        clients = backend.backends()
        selected = backend.get_backend()
        platform = next(name for name, client in clients.items() if client is selected)
        if platform not in GPU_VISIBLE_DEVICE_ENV:
            raise RuntimeError(
                f"train_dist requires a CUDA or ROCm GPU; JAX selected {platform!r}. "
                "Check your JAX GPU installation and device visibility settings."
            )
        devices = [
            {
                "local_hardware_id": device.local_hardware_id,
                "device_kind": device.device_kind,
            }
            for device in selected.local_devices()
        ]
        if not devices:
            raise RuntimeError(f"JAX backend {platform!r} has no local GPUs")
    return {"platform": platform, "devices": devices}


if __name__ == "__main__":
    print(json.dumps(_probe()))
