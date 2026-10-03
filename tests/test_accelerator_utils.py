"""GPU discovery and visibility-mask checks."""

import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from scripts import accelerator_utils


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_probe_uses_registered_backend_identity(monkeypatch, platform):
    selected = MagicMock()
    selected.local_devices.return_value = [
        SimpleNamespace(local_hardware_id=2, device_kind="GPU model")
    ]
    extension = ModuleType("jax.extend")
    extension.backend = SimpleNamespace(
        backends=lambda: {"cpu": MagicMock(), platform: selected},
        get_backend=lambda: selected,
    )
    monkeypatch.setitem(sys.modules, "jax.extend", extension)
    assert accelerator_utils._probe() == {
        "platform": platform,
        "devices": [{"local_hardware_id": 2, "device_kind": "GPU model"}],
    }


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_binding_preserves_runtime_masks_and_parent_environment(monkeypatch, platform):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,5")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    probe = {
        "platform": platform,
        "devices": [
            {"local_hardware_id": 0, "device_kind": "first"},
            {"local_hardware_id": 1, "device_kind": "second"},
        ],
    }
    monkeypatch.setattr(accelerator_utils, "probe_gpus", lambda: probe)
    original = dict(os.environ)
    settings, device, num_gpus = accelerator_utils.gpu_visibility_settings(job_id=3)
    assert num_gpus == 2
    assert device["device_kind"] == "second"
    assert settings[accelerator_utils.GPU_VISIBLE_DEVICE_ENV[platform]] == "1"
    assert settings["JAX_PLATFORMS"] == platform
    assert set(settings) == {
        "JAX_PLATFORMS",
        accelerator_utils.GPU_VISIBLE_DEVICE_ENV[platform],
    }
    assert dict(os.environ) == original


def test_binding_rejects_empty_device_pool():
    with pytest.raises(ValueError, match="No visible"):
        accelerator_utils._gpu_visibility_settings(
            {"platform": "cuda", "devices": []}, 0
        )


def test_probe_failure_keeps_gpu_diagnostics(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="GPU driver failed")

    monkeypatch.setattr(accelerator_utils.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="GPU driver failed"):
        accelerator_utils.probe_gpus()


def test_real_cpu_probe_does_not_import_jax_in_caller():
    code = (
        "import sys; from scripts.accelerator_utils import probe_gpus; "
        "\ntry: probe_gpus()"
        "\nexcept RuntimeError as error: assert \"selected 'cpu'\" in str(error)"
        "\nelse: raise AssertionError('CPU must be rejected')"
        "\nassert 'jax' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
