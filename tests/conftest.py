import gc
import math
import os

import pytest


@pytest.fixture(params=["gymnasium", "envpool"])
def host_envs(request, monkeypatch, tmp_path):
    """Track native host environments and close their worker resources."""
    # Native simulators can write logs/configuration in the cwd.
    monkeypatch.chdir(tmp_path)
    backend = request.param
    library = pytest.importorskip(backend)
    factory_name = "make_vec" if backend == "gymnasium" else "make_gymnasium"
    original_factory = getattr(library, factory_name)
    environments = []

    def make_env(*args, **kwargs):
        if backend == "envpool":
            kwargs.setdefault("num_threads", 2)
        env = original_factory(*args, **kwargs)
        environments.append(env)
        return env

    monkeypatch.setattr(library, factory_name, make_env)
    try:
        yield backend, environments
    finally:
        import jax

        try:
            jax.effects_barrier()
        finally:
            for env in environments:
                env.close()


def pytest_addoption(parser):
    parser.addoption(
        "--jumanji-catalog",
        action="store_true",
        help="Check every Jumanji YAML under all four autoreset modes and nested vmap.",
    )
    parser.addoption(
        "--host-env-catalog",
        action="store_true",
        help="Run every EnvPool YAML through JIT and nested vmap instead of family representatives.",
    )
    parser.addoption(
        "--host-env-stress",
        action="store_true",
        help="Run large native environment batches in addition to regression tests.",
    )
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Choose the device to run tests on: cpu or gpu",
    )
    parser.addoption(
        "--cpu-devices",
        type=int,
        default=2,
        help="Number of logical CPU devices initialized before importing JAX.",
    )


def pytest_configure(config):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_DEBUG_NANS"] = "true"
    if config.getoption("--device") == "cpu":
        count = config.getoption("--cpu-devices")
        if count < 1:
            raise pytest.UsageError("--cpu-devices must be positive")
        os.environ["JAX_NUM_CPU_DEVICES"] = str(count)
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_deterministic_ops=true".strip()


@pytest.fixture(params=[True, False], ids=["jit", "eager"])
def jit(request):
    return request.param


@pytest.fixture
def mesh_factory():
    import jax

    def make(shape=(2,), names=("DP",)):
        size = math.prod(shape)
        devices = jax.local_devices()
        if len(devices) < size:
            pytest.skip(f"Requires {size} devices; run with --cpu-devices={size}")
        return jax.make_mesh(
            shape,
            names,
            axis_types=(jax.sharding.AxisType.Explicit,) * len(shape),
            devices=devices[:size],
        )

    return make


@pytest.fixture
def dp_mesh(mesh_factory):
    import jax

    mesh = mesh_factory()
    with jax.set_mesh(mesh):
        yield mesh


@pytest.fixture
def meta_mesh(mesh_factory):
    import jax
    from evorl.distributed import META_POP_AXIS_NAME

    mesh = mesh_factory(names=(META_POP_AXIS_NAME,))
    with jax.set_mesh(mesh):
        yield mesh


@pytest.fixture(autouse=True)
def release_integration_compilations(request):
    yield
    if request.node.get_closest_marker("integration"):
        import jax

        jax.clear_caches()
        gc.collect()
