# Contributing to EvoRL

## Install Developer Version

```shell
# Step 1: Create a virtual env from conda, venv, pyenv, ...:
# For example, using pyenv-virtualenv:
pyenv virtualenv 3.12 evorl-dev
pyenv activate evorl-dev

# Step 2: Install JAX in advance
# Note: the installation command depends on your platform.
# See https://docs.jax.dev/en/latest/installation.html
pip install -U "jax[cuda12]"

# Step 3: Install the developer version
pip install -e ".[dev]"

# Step 4: post-setup
# This repo is configured with `pre-commit`.
pre-commit install
```

Aim is included in the developer installation. To work on an optional tracking
backend, combine its extra with `dev`, for example `pip install -e ".[dev,wandb]"`,
or install the SDK directly. See [Experiment Logging installation](../guide/installation.md#experiment-logging)
for the available extras and package names.

### Dependency compatibility

The framework requires Python 3.12+, JAX/jaxlib 0.7.1+, and Distrax 0.1.9+.
JAX 0.7.1 satisfies Distrax's requirements and adds `jax.set_mesh`, which
allows the sharding APIs used here to accept raw `jax.P` specs.
Python and JAX minimum versions are independent.

CI tests the built wheel with both JAX/jaxlib 0.7.1 + Orbax 0.11.30 and the
latest dependencies. The minimum job lets the resolver select compatible
versions of Flax and other dependencies; the latest Flax release can require
a newer JAX version. The framework continues to use Flax Linen and the Orbax
checkpoint V0 API.

Orbax requires version 0.11.30 or newer for the typed-key checkpoint restore
path exercised by our compatibility suite.
See the [Orbax changelog](https://github.com/google/orbax/blob/main/checkpoint/CHANGELOG.md).

The compatibility audit on 2026-10-09 covers these combinations:

| Component | Minimum JAX combination | Latest combination |
| --- | --- | --- |
| Python | 3.12 | 3.12 |
| JAX / jaxlib | 0.7.1 | 0.11.2 |
| Flax (Linen) | 0.11.2 | 0.12.10 |
| Orbax (V0) | 0.11.30 | 0.12.7 |
| Optax | 0.2.8 | 0.2.8 |
| Chex | 0.1.90 | 0.1.92 |
| Distrax | 0.1.9 | 0.1.9 |
| NumPy / SciPy / pandas | 2.5.3 / 1.18.1 / 3.0.6 | 2.5.3 / 1.18.1 / 3.0.6 |
| TFP nightly | 0.26.0.dev20261008 | 0.26.0.dev20261008 |

These are tested combinations, not independent lower bounds for every package.
Coverage includes typed-key batching and host callbacks, Linen initialization,
mutable normalization state and gradients, seven RL agents, Optax hyperparameter
updates, running statistics, two-device CPU sharding and gradients, V0
checkpoints, and logging with NumPy/pandas.

DQN uses Distrax's `EpsilonGreedy` distribution; other distributions use
TFP's JAX substrate. Distrax 0.1.9+ depends on `tfp-nightly`, matching the
existing direct dependency. Do not install stable `tensorflow-probability`
alongside `tfp-nightly`: they install overlapping files.

Construct device meshes with `jax.make_mesh` and inline simple `jax.P` specs.
Keep named specs for structured state trees or generic function parameters.
Training meshes use `axis_types=(jax.sharding.AxisType.Explicit,)` explicitly
so JAX 0.7.1 and newer versions use the same mode. RL stores a replicated scalar
root key. Each iteration splits the next root from the keys it consumes, then
folds `lax.axis_index(dp_axis_name)` into the consumed keys for independent local
random streams. Never fold the saved next root: it must remain replicated.
Environment setup, evaluation and buffer warmup follow the same rule; model
initialization keeps a shared key so parameters start identical. Single-device paths
keep their original split sequences; distributed random sequences change.
EC retains its worker-key batch; PBT generates `pop_size` keys and shards the
population batch. Ordinary RL uses one worker per device; PBT may assign multiple
individuals to each device. RL environment arrays have global shape
`(total_num_envs, ...)` and are partitioned along that existing environment axis.
Each device resets and steps `(local_num_envs, ...)` using the environment's
existing batching; do not add an outer worker `vmap` to the batched environment.
PBT uses its existing inner `vmap` to process local population members.
`shard_map` inputs must already match their declared specs.
RL model parameters, metrics and root keys are replicated; environments and
per-device replay buffers are sharded. PBT replicates fitnesses for global selection and
selected parents for mutation, then writes updates to their owning shards.
Full model populations stay sharded. Each distributed workflow stores one
`mesh`, without a separate device list or `NamedSharding` object. Training
entry points enter `with jax.set_mesh(workflow.mesh):` before initialization
and learning. Direct callers must enter the same context before setup, step,
or evaluation (including before calling jitted versions). `jax.shard_map`
inherits that mesh; placement and gather operations use raw `jax.P` specs.
Never call `jax.set_mesh` inside a jitted function.

The outer mesh uses Explicit axes; `jax.shard_map` enters Manual mode for
its body and returns to the outer mode afterward. Manual mode is appropriate
for device-local rollout/update code and explicit collectives; it does not
mean there is no communication. DP synchronizes gradients with `pmean` and
reduces metrics/running statistics. EC and shared-buffer PBT gather data.
Do not create the outer mesh with Manual axes: `jax.set_mesh` rejects them.
Build each execution configuration on a fresh workflow subclass, as in
`scripts/train.py`. `build_from_config(..., multi_device=True, jit=True)` creates
the mesh, applies `enable_shmap`, then applies `enable_jit`. Both hooks are class
methods. The construction flag is `multi_device`; the instance predicate is
`is_multi_device`, derived from whether `dp_axis_name` is set. Workflow branches
use this predicate rather than checking the axis name directly.
Computation reads fixed workflow attributes; changing training data
is passed explicitly through `State`. `shard_map` declares static `self` with
`in_specs=(None, state_specs)`, and `jit` uses `static_argnums=(0,)`.
Concrete workflow templates write their state layouts directly. Parameters and
metrics retain their replicated shapes. Environment arrays pass directly between
setup, step and evaluate, with no device dimension or squeeze/expand operations.
Replay buffers remain independent local rings. Data and priorities have global
shape `(total_capacity, ...)`, sliced into `(local_capacity, ...)` on each device.
Sample indices remain local indices; their batches are concatenated on output.
Only per-ring scalar fields (`current_index`, `buffer_size`, and prioritized
buffers' `max_priority`) use one value per device and explicit scalar conversion
at the method boundary. `_shard_map_replay_buffer_context` in `rl_workflow.py`
yields a shallow working copy of `State` with these fields squeezed on entry.
Callers use `local_state.update(next_state)` to retain the method's returned state;
normal exit expands its scalar fields before the caller returns `local_state`.
Only the working copy is mutated, never the input state. The context is shared
by training and replay-buffer warmup.
These fields must not be marked replicated: masked insertion can
write different numbers of transitions on different devices. Evaluation passes
the buffer through unchanged. Computation runs in Manual mode across the data-parallel
axis, with communication expressed by explicit collectives. Keep the State's
input and output shardings identical to avoid resharding between calls; `P()`
on an output is a replication promise, not an implicit reduction.
Loss metrics must be explicitly reduced as well as gradients before declaring
them replicated. Tests compare actual replica values and inspect compiled
collectives: PPO/SAC use all-reduce, and a local-only computation has none.
The workflow tests cover distributed initialization and PBT shard-map paths,
plus PPO, SAC, OpenES and RandomAgent setup/step/evaluate on two CPU devices.
Uniform, prioritized and LAP replay tests cover unequal masked insertion, empty
writes, wraparound, local sampling and priority updates against independent local
references. Their compiled buffer steps must contain no collective communication.
Current JAX supports these checks with and without outer JIT. On JAX 0.7.1,
use `jit=True`: eager `shard_map` can fail with a gather `ShardingTypeError`
on `fold_in(key, axis_index)` or replicated arrays captured by `scan`, independently
of Workflow or static `self`. RL's `enable_jit` compiles setup as well as step
and evaluation, so initialization uses the same compiled execution mode.
The eager regression tests record this failure as xfail only on 0.7.1.


TFP nightly still uses deprecated JAX internals, and Brax imports unmaintained
JAXopt. The tested versions work; future upstream changes need CI verification.
Optional RL environment upgrades and multi-host GPU training are outside this
audit's test coverage. Tests must set `XLA_PYTHON_CLIENT_PREALLOCATE=false`; the
compatibility suite runs on CPU.

### Type annotations

Use `jax.Array` for JAX array fields, outputs, and typed PRNG keys.
Use `jax.typing.ArrayLike` for inputs that also accept NumPy arrays or numeric
scalars; this annotation does not promise an input already has array attributes.
Use `jax.typing.DTypeLike` for dtype arguments and `tuple[int, ...]` for shapes.
Keep `chex.ArrayTree` for array pytrees and Chex assertions where needed.
See [JAX typing](https://docs.jax.dev/en/latest/jax.typing.html).

A `jax.Array` key annotation does not encode its PRNG dtype or batch shape:
key helpers validate the dtype, and callers construct keys with `jax.random.key`.

### Random keys and checkpoints

Construct new keys with `jax.random.key(seed)`. A single typed key has `shape ()`,
and a batch has `shape (B,)`, with the RNG implementation carried by its dtype.
Use `rng_split` for possibly batched keys, `rng_split_by_shape` for a new key batch,
and `jax.random.key_data` only at host callback boundaries that require NumPy data.
See [JAX typed keys](https://docs.jax.dev/en/latest/jep/9263-typed-keys.html).

Checkpoint restores assume the current state structure and typed-key
representation. We do not maintain checkpoint format compatibility or add
migration code for old checkpoints. For an abstract restore template, use
`jax.tree.map(evorl.utils.orbax_utils.to_shape_dtype_struct, state)` to preserve
typed-key shapes and dtypes; Orbax's general conversion helper expands keys to
their raw uint32 representation.

Deferred audits are tracked in [Development TODO](todo.md).

### Debugging

We provide some examples in `.vscode/launch.json` for debugging in Visual Studio Code.

## Documentation

Documentations are written under the `docs/` directory. We use [`Sphinx`](https://www.sphinx-doc.org/) to construct the documentation. Since we use the [`MyST-Parser`](https://myst-parser.readthedocs.io/) extension, Markdown documents (`*.md`) all allowed. API References are automatically generated by [`autodoc2`](https://sphinx-autodoc2.readthedocs.io/).

If you want to build the documentations in local machine, run the script `./docs/build.sh`. Then, the documentation Web files are generated in `docs/_build`.

## Code Style

In general, we follow the [Black code style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) for Python code. For Python documentation, we follow the [Google pydoc format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

The following code snippets demonstrate the code style in EvoRL:

```python
class Workflow(AbstractWorkflow):
    """The base class for all Workflows.

    All workflow classes are inherit from this class, and customize by implementing
    """

    def __init__(self, config: DictConfig):
        """Initialize a RLWorkflow instance.

        Args:
            config: the config object.
        """
        self.config = config
        self.recorder = ChainRecorder([])
        self.checkpoint_manager = setup_checkpoint_manager(config)
```

```python
class ReplayBufferState(PyTreeData):
    """Contains data related to a replay buffer.

    Attributes:
        data: the stored replay buffer data.
        current_index: the pointer used for adding data.
        buffer_size: the current size of the replay buffer.
    """

    data: chex.ArrayTree
    current_index: jax.Array = jnp.zeros((), jnp.int32)
    buffer_size: jax.Array = jnp.zeros((), jnp.int32)
```

Developers do not need to worry about whether the submitted code satisfy the above code style. Before submitting any commit, the `pre-commit` will raise warning or errors to remind you what is wrong. Besides, developers can manually execute this process by running

```shell
pre-commit run -a
```


## Publishing a Release

EvoRL is distributed on PyPI as `evorl-jax` and imported in Python as `evorl`.
The project is in beta: the implemented algorithms are usable, while APIs and
configuration may still change between releases. The version is maintained in
`evorl/__init__.py`; setuptools reads `evorl.__version__` when building packages.

### Prepare the release commit

1. Finish the release changes on `main` and update `evorl.__version__`. For the
   first release, the version is `0.1.0`.
2. Review changes since the previous release, including new features, bug fixes,
   dependency requirements, and any API or configuration changes users need to make.
3. Validate the packages from a clean output directory:

   ```shell
   python -m pip install build twine
   python -m build --outdir /tmp/evorl-release-dist
   python -m twine check --strict /tmp/evorl-release-dist/*
   python scripts/check_dist.py --dist-dir /tmp/evorl-release-dist --tag v0.1.0
   ```

   Use an empty output directory for each build; old release artifacts must not
   be included in the upload. Update the tag in the example for subsequent versions.
4. Commit the reviewed release changes, push `main`, and wait for branch CI to pass.
   The PyPI Trusted Publisher and GitHub `pypi` environment must already be configured
   for `.github/workflows/release.yml` before pushing a release tag.

### Tag and publish

An annotated tag (`git tag -a`) marks a release commit and stores its author,
date, and message. The message describes the whole version, independently of
commit messages. Write the release summary in `-m`, then push the tag:

```shell
git switch main
git status --short  # Release changes must already be committed.
git tag -a v0.1.2 --cleanup=whitespace -m '## Highlights
- Summarize new features and fixes.

## Compatibility
- Describe breaking changes or migration steps; omit if unnecessary.'
git push origin v0.1.2
```

Replace the example version and summary before running. `--cleanup=whitespace`
preserves Markdown headings beginning with `#`. The tag must match
`evorl.__version__`: `v0.1.2` requires `0.1.2`. Release candidates use a version
such as `0.2.0rc1` and tag `v0.2.0rc1`, marked as a GitHub prerelease.

Tag pushes run CI against the tagged commit, then publish the wheel and sdist to
PyPI and GitHub Release in parallel. Pushing `main` alone only runs CI.

### Release description

The workflow uses the annotated tag message as the GitHub Release summary and
appends GitHub-generated change links. Without a tag annotation, it uses only
generated notes. Focus the summary on user-visible changes and compatibility.

After publishing, use **Releases → version → Edit** to refine the description.
Retries preserve these edits. GitHub Release text does not update PyPI metadata;
PyPI displays the packaged README and a **Changelog** link to GitHub Releases.

### Run an existing tag manually

If a tag already exists but its release workflow never started, open **Actions →
Release → Run workflow**, choose `main` for the workflow, and enter the existing
version tag (for example `v0.1.0`) in the `tag` field. The workflow on `main`
resolves and builds the tagged commit rather than building the latest `main`.
This does not create, delete, or move a tag. Normal releases still start with
`git push origin vX.Y.Z`.

### Handle a failed release

If a publishing job fails, correct its configuration and rerun only the failed
jobs. The PyPI and GitHub Release jobs run independently: retrying a failed PyPI
job does not recreate the GitHub Release, and retrying a failed GitHub job does not
upload the package to PyPI again. The GitHub job updates distribution assets from
the current run when the release already exists. If any PyPI files have already uploaded,
check the published files before retrying; an upload can fail after partial success.

PyPI files cannot be replaced for an already uploaded version. If a release needs
code changes, update the version and publish a new tag. Do not move or reuse a
published version tag. Editing the GitHub Release description is safe and does
not change the uploaded package.
