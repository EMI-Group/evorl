import pytest
from .utils import disable_gpu_preallocation


@pytest.fixture(autouse=True, scope="session")
def run_before_and_after_tests():
    print("turn off jax GPU preallocation")

    disable_gpu_preallocation()
    yield
