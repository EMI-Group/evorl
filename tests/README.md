# Tests

The JAX and workflow regression tests are ordinary pytest tests. Each algorithm,
buffer type and execution mode has its own test node and assertion traceback.

| Location | Coverage |
| --- | --- |
| `distributed/` | Mean loss/gradients, local auxiliary outputs, optimizer updates, collectives, local maps and VMA casts |
| `replay_buffers/` | Independent device-local rings, masked/empty writes, wraparound, sampling and priorities |
| `workflows/test_rl.py` | Real RL/OpenES/Random workflow setup, step, evaluation, keys and state layouts |
| `workflows/test_pbt.py` | Independent PBT members, dynamic rollout termination and shared replay initialization |
| `workflows/test_execution.py` | Template state layouts, population selection/transfer and per-member statistics |
| Other `workflows/` tests | Training budgets, final evaluation/checkpoints, warmup and EC accounting |
| `helpers/` | Reusable toy environment and compiled-communication inspection |
| `test_host_envs.py` | Ordered host effects across resets, constant actions, nested `vmap`, autoreset modes and Atari RGB/RAM configuration |
| `test_host_env_shapes.py` | Scalar/matrix actions, array/nested Dict observations, terminal observations, nested `vmap` and unsupported spaces/modes |
| `test_envpool_catalog.py` / `test_gymnasium_catalog.py` | Configured task shapes/dtypes under JIT, including nested EnvPool populations |
| `test_host_env_stress.py` | Opt-in nested batches with up to 65,536 EnvPool or 16 Gymnasium environments |
| `workflows/test_host_envs.py` | Real PPO training/evaluation and nested PBT/OpenES populations with both host backends |
| `workflows/test_eval_batching.py` | Pure JAX and host early-stop metrics/final states, mixed shared/mapped agent leaves and keys |
| `test_brax_env.py::test_brax_fast` | Brax Fast action bounds, array/dict observations and all autoreset modes |
| `test_gymnax_env.py` | Constructor arguments, `EnvParams` overrides, unknown argument errors and EvoRL episode limits |
| Root `test_*.py` files | Existing unit, dependency, checkpoint, environment and algorithm tests |

Install the development dependencies with `pip install -e ".[dev]"`.
Run the distributed and workflow suite from the repository root:

```shell
python -m pytest tests/distributed tests/replay_buffers tests/workflows -m "not four_devices"
```

Tests default to CPU with two logical devices. `conftest.py` configures the device
count before test collection imports JAX and disables GPU preallocation. The
two-axis VMA test needs a separate process with four devices:

```shell
python -m pytest tests/distributed --cpu-devices=4 -m four_devices
```

Run host workflows on one CPU device or a GPU, with optional large batches:

```shell
python -m pytest tests/test_host_envs.py tests/workflows/test_host_envs.py --cpu-devices=1
python -m pytest tests/test_host_envs.py tests/workflows/test_host_envs.py --device=gpu
python -m pytest tests/test_host_env_stress.py --host-env-stress --device=gpu
python -m pytest tests/test_envpool_catalog.py tests/test_gymnasium_catalog.py --host-env-catalog
```

Host simulators run on the CPU even when the policy runs on the GPU. PBT host
workflow tests require one JAX device because native pools do not support
splitting across devices. Stress tests are skipped unless explicitly enabled.
Catalog tests normally select family representatives; `--host-env-catalog`
executes every EnvPool configuration and current Gymnasium configuration.
Gymnasium MuJoCo v2/v3 tasks use legacy `mujoco-py` and are excluded from this
catalog. Optional simulator dependencies and assets must be installed.

The `jit` fixture gives supported tests `jit` and `eager` cases. Select a single
algorithm or regression using pytest's normal file, node and `-k` options:

```shell
python -m pytest tests/workflows/test_rl.py -k "ppo and jit"
python -m pytest tests/workflows/test_warmup.py -k pbt
python -m pytest tests/distributed/test_gradients.py
```

Workflow construction changes class methods, so tests create fresh subclasses
for each execution configuration. Tests marked `integration` release JAX
compilation caches after each case to bound memory when several algorithms run
in one process. Mesh fixtures restore their surrounding context on exit.

Communication assertions inspect compiled collectives, not instruction names,
exact HLO text or wall-clock timing. Gradient tests compare analytical values;
replay tests compare independent local ring references; PBT tests compare member
updates and a single-device population reference. Keep subprocesses only when
the process boundary itself is under test, such as importing accelerator probes.

CI runs the same collected tests against the built wheel with the minimum and
latest supported dependencies, including the separate four-device invocation.
The host environment tests run in CI with Gymnasium; EnvPool and Atari cases run
when their optional dependencies are installed. Other environment-specific tests
in the root directory may require simulator dependencies and assets and remain
outside this CPU suite.
