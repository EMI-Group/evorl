import pytest
from .utils import (
    disable_gpu_preallocation,
    enable_deterministic_mode,
    enable_nan_inf_check,
)


@pytest.fixture(autouse=True, scope="session")
def run_before_and_after_tests():
    print("turn off jax GPU preallocation")

    disable_gpu_preallocation()
    enable_nan_inf_check()
    enable_deterministic_mode()
    yield
