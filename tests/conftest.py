import os


def set_default_device_cpu():
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def disable_gpu_preallocation():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def enable_nan_inf_check():
    os.environ["JAX_DEBUG_NANS"] = "true"


def enable_deterministic_mode():
    xla_flags = os.getenv("XLA_FLAGS", "")
    print(f"current XLA_FLAGS: {xla_flags}")
    if len(xla_flags) > 0:
        xla_flags = xla_flags + " "
    os.environ["XLA_FLAGS"] = xla_flags + "--xla_gpu_deterministic_ops=true"


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Choose the device to run tests on: cpu or gpu",
    )


def pytest_configure(config):
    device = config.getoption("--device")
    enable_nan_inf_check()
    if device == "cpu":
        print("Use CPU!")
        set_default_device_cpu()
    else:
        print("Use GPU!")
        print("Turn off jax GPU preallocation!")
        disable_gpu_preallocation()
        enable_deterministic_mode()
