import os


def disable_gpu_preallocation():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
