import pytest
from .utils import (
    disable_gpu_preallocation,
    enable_deterministic_mode,
    enable_nan_inf_check,
    set_default_device_cpu,
)

def pytest_addoption(parser):
    parser.addoption(
        "--device", action="store", default="cpu", choices=["cpu", "gpu"],
        help="Choose the device to run tests on: cpu or gpu"
    )

@pytest.fixture(autouse=True, scope="session")
def run_before_and_after_tests(request):
    device = request.config.getoption("--device")

    enable_nan_inf_check()
    if device == "cpu":
        print("Use CPU!")
        set_default_device_cpu()
    else:
        print("Use GPU!")
        print("Turn off jax GPU preallocation!")
        disable_gpu_preallocation()
        enable_deterministic_mode()
    yield
