# Tests

The JAX and workflow regression tests are ordinary pytest tests. Each algorithm,
buffer type and execution mode has its own test node and assertion traceback.

| Location | Coverage |
| --- | --- |
| `networks/` | Native NNX MLP/CNN shapes, ensembles, normalization state, gradients and target updates |
| `agents/` | Configuration-driven NNX initialization, selected-network and joint updates versus JAX/Optax references, input/target isolation, nested populations and dynamic PBT hyperparameters |
| `agents/test_impala.py` | V-trace clipping against a NumPy recurrence and on-policy GAE equivalence, including termination and truncation |
| `workflows/test_offpolicy.py` | Full TD3/TD7/DDPG/SAC/DQN updates versus explicit parameter gradients, update schedules, target copies and forward-variable propagation through state-based losses |
| `workflows/test_erl.py` | Population actors with shared or independent critics and temporary actor optimizer state |
| `workflows/test_pop_td3.py` | Independent TD3 members, shared replay, observation statistics and optimizer update counts |
| `distributed/` | Mean loss/gradients, local auxiliary outputs, optimizer updates, collectives, local maps and VMA casts |
| `replay_buffers/` | Independent device-local rings, masked/empty writes, wraparound, sampling and priorities |
| `workflows/test_rl.py` | Real RL/OpenES/Random workflow setup, step, evaluation, keys and state layouts |
| `workflows/test_pbt.py` | Independent PBT members, optimizer learning-rate mutation, dynamic rollout termination and shared read-only replay training |
| `workflows/test_execution.py` | Template state layouts, population selection/transfer and per-member statistics |
| Other `workflows/` tests | Training budgets, final evaluation/checkpoints, warmup and EC accounting |
| `helpers/` | Reusable toy environment and compiled-communication inspection |
| `test_env_factory.py` | YAML argument forwarding, episode-limit defaults and explicit override precedence across environment factories |
| `ec/test_operators.py` | MLP mutation counts/bounds, parent-row inheritance, LayerNorm preservation and population keys |
| `ec/test_optimizers.py` | CMA-ES reference equations/convergence, NSGA-II elitism/ZDT1, nested `vmap` and replicated execution |
| `ec/test_non_dominated_sort.py` | Independent Pareto references, all sorting modes, ties, nested batches, backend dispatch and linear scratch memory |
| `ec/test_param_vector.py` | Parameter vector/tree roundtrips with Flax, mixed dtypes, scalar leaves and empty/nested batches |
| `workflows/test_ec.py` | Native EC setup, updates, evaluation, logging, final checkpoints and single/multiple device execution |
| `evox/test_algorithm.py` | Explicit tensor/RNG state, replay, failure isolation, DLPack, dtype checks and native RVEA equivalence |
| `evox/test_workflow.py` | Initialization budgets, scalar/vector rewards, independent JAX/Torch compilation, evaluation counts and checkpoint resume |
| `evox/test_brax.py` | Real CSO/RVEA training, observation warmup, file logging and checkpoint replay; full-length cases carry `slow` |
| `test_torch_utils.py` | Optional PyTorch: DLPack conversion and storage isolation/sharing, dtype/device preservation, CPU/GPU device mapping, local hardware indices, RNG replay and restoration on success/failure |
| `test_host_envs.py` | Ordered host effects across resets, constant actions, nested `vmap`, autoreset modes and Atari RGB/RAM configuration |
| `test_host_env_shapes.py` | Scalar/matrix actions, array/nested Dict observations, terminal observations, nested `vmap` and unsupported spaces/modes |
| `test_envpool_catalog.py` / `test_gymnasium_catalog.py` | Configured task shapes/dtypes under JIT, including nested EnvPool populations |
| `test_host_env_stress.py` | Opt-in nested batches with up to 65,536 EnvPool or 16 Gymnasium environments |
| `workflows/test_host_envs.py` | Real PPO training/evaluation and nested PBT/OpenES populations with both host backends |
| `workflows/test_eval_batching.py` | Pure JAX and host early-stop metrics/final states, mixed shared/mapped agent leaves and keys |
| `test_brax_env.py::test_brax_fast` | Brax Fast action bounds, array/dict observations and all autoreset modes |
| `test_gymnax_env.py` | Configured environments under JIT, constructor and `EnvParams` overrides, unknown argument errors, and nested vmap across episode resets in all four autoreset modes |
| `test_multi_agent_env.py` | Shared environment/wrapper interfaces, individual and whole-episode boundaries, per-agent returns, terminal/global observations and nested autoreset |
| `test_jaxmarl_env.py` | Configured native transitions, per-agent spaces/rewards, legal action masks, global observations and all autoreset modes under nested `vmap` |
| `test_vector_rewards.py` | Scalar and vector reward shapes, terminal masking, nested evaluation, action repeat, reward scaling, autoreset and native Brax/Playground metric selection |
| `test_jumanji_env.py` | Native state conversion, structured observations, extras and action spaces |
| `test_jumanji_multi_agent.py` | Native states/rewards, discount-to-done conversion, native timeouts, joint actions, per-agent spaces, masks and observations for all seven multi-agent tasks |
| `test_jumanji_autoreset.py` | EvoRL wrapper selection, staggered episodes and reset keys, terminal masks/observations, shared native time limits, termination/truncation and frozen evaluation states |
| `test_jumanji_catalog.py` | Registered spaces, YAML tasks under nested `vmap`, serial references and in-memory Sokoban levels |
| `workflows/test_jumanji_env.py` | PPO training/evaluation with dictionary observations and RandomAgent evaluation with integer vector actions |
| Root `test_*.py` files | Existing unit, dependency, checkpoint, environment and algorithm tests |

Install the development dependencies with `uv pip install -e ".[dev]"`.
Run the distributed and workflow suite from the repository root:

```shell
python -m pytest tests/networks tests/agents tests/ec tests/distributed tests/replay_buffers tests/workflows -m "not four_devices"
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

With the Jumanji extra installed, check its registered environments, episode
wrappers and PPO integration with:

```shell
python -m pytest tests/test_multi_agent_env.py tests/test_jumanji_env.py tests/test_jumanji_multi_agent.py tests/test_jumanji_autoreset.py tests/test_jumanji_catalog.py tests/workflows/test_jumanji_env.py
python -m pytest tests/test_jumanji_catalog.py --jumanji-catalog
```

The Jumanji catalogue checks every YAML with next-step autoreset, plus all four
autoreset modes for representative tasks. `--jumanji-catalog` checks every YAML
under all four modes. Each nested population is compared with separate serial
runs using the same keys. Sokoban uses upstream in-memory levels in tests; its
default external dataset is not downloaded.

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

CI runs these suites against the built wheel from outside the checkout.
The core job includes the entire `ec/` and `workflows/` directories, with
minimum and latest JAX dependencies and a separate four-device invocation.
A separate EvoX job installs the optional extra with CPU PyTorch and tests
minimum JAX/PyTorch 2.9 and latest dependencies.

Run the optional interoperability suite or full-length Brax training with:

```shell
uv pip install -e ".[dev,evox]"
python -m pytest tests/evox tests/test_torch_utils.py -m "not slow"
python -m pytest tests/evox/test_brax.py --device=gpu -m slow
```

The normal EvoX CPU suite includes short real-Brax evaluations. Full-length
training tests remain ordinary pytest cases marked `slow`; CPU CI excludes
those four cases. The marker does not skip them in an unfiltered local run.
Shared EvoX test constructors live in `helpers/evox.py`; shared scalar/vector
workflow environments live in `helpers/envs.py`. Test modules do not import
other test modules.

Brax and MuJoCo Playground are required dependencies. Jumanji, JaxMARL and
Gymnasium environment tests also run in CI; EnvPool and Atari cases run when
their optional dependencies are installed. Other simulator-dependent root
tests remain available for local runs.
