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

Use Google-style docstrings with a concise summary and `Args:`, `Returns:`,
and `Raises:` sections where appropriate. Use single backticks for inline code
and identifiers, matching this project's MyST documentation rendering.

### Dependency compatibility

The framework requires Python 3.12+, JAX/jaxlib 0.10.0+, and Distrax 0.1.9+.
JAX 0.10.0 provides `jax.lax.all_gather(..., to="invarying")` and the manual-axis
types used by the multi-device workflows' strict VMA checks.
Python and JAX minimum versions are independent.

CI tests the built wheel with both JAX/jaxlib 0.10.0 + Orbax 0.11.30 and the
latest dependencies. The minimum job lets the resolver select compatible
versions of Flax and other dependencies; the latest Flax release can require
a newer JAX version. Networks use Flax NNX 0.12.8+; checkpoints use the Orbax
V0 API.

Orbax requires version 0.11.30 or newer for the typed-key checkpoint restore
path exercised by our compatibility suite.
See the [Orbax changelog](https://github.com/google/orbax/blob/main/checkpoint/CHANGELOG.md).

The compatibility suite covers typed-key batching and host callbacks, NNX
initialization, mutable normalization state and gradients, RL agents,
Optax hyperparameter updates, running statistics, two-device CPU sharding and
gradients, V0 checkpoints, and logging with NumPy/pandas.

DQN uses Distrax's `EpsilonGreedy` distribution; other distributions use
TFP's JAX substrate. Distrax 0.1.9+ depends on `tfp-nightly`, matching the
existing direct dependency. Do not install stable `tensorflow-probability`
alongside `tfp-nightly`: they install overlapping files.

### Agent state and NNX training

Agents hold fixed configuration. `AgentState.model_state` holds a complete
`nnx.GraphState`: its graph definition is static PyTree metadata, while network
parameters and auxiliary variables are dynamic leaves. Agent loss methods take
`AgentState` and return a loss dictionary; workflows choose which parameters
to differentiate and apply optimizer updates.

For example, TD3's critic update uses this sequence:

```python
models = nnx.merge(agent_state.model_state, copy=True)

def loss_fn(models, batch, key):
    working_state = agent_state.replace(model_state=nnx.unpack(models))
    losses = agent.critic_loss(working_state, batch, key)
    return losses.critic_loss, losses

value, grads = pmean_nnx_loss_and_grad(
    loss_fn,
    axis_name,
    has_aux=True,
    wrt=nnx.All(nnx.Param, nnx.PathContains("critic")),
)(models, batch, key)
optimizer.update(models.critic, grads["critic"])
next_agent_state = agent_state.replace(model_state=nnx.unpack(models))
```

`nnx.unpack(models)` retains references to the existing NNX variables.
`Agent.get_models(working_state)` calls `nnx.merge(..., copy=False)`: it creates
new module objects sharing those variables with the working `models`.
Consequently, a forward-pass write such as `variable[...] = value` reaches the
models managed by `nnx.value_and_grad`. This includes non-parameter state such
as SpectralNorm's singular vector, even though only critic parameters receive
gradients. The final snapshot includes both the optimizer's parameter updates
and these forward-pass variable updates; the loss need not return another state.

The initial `copy=True` isolates variable objects from the input snapshot.
Rollout and evaluation also use isolated models so their input state is preserved.
For the array-backed variables used here, these copies create new Python
variable objects without duplicating device array buffers.
Module structure is established during `Agent.init()`: the shared-variable view
propagates updates to existing variable values, not replacement of module
attributes with new variables or submodules.

### Multi-device execution

Construct device meshes with `jax.make_mesh` and inline simple `jax.P` specs.
Keep named specs for structured state trees or generic function parameters.
Training meshes use `axis_types=(jax.sharding.AxisType.Explicit,)` explicitly.
RL stores a replicated scalar root key. Each iteration splits the next root
from the keys it consumes, then folds `lax.axis_index(dp_axis_name)` into the
consumed keys for independent local random streams. Never fold the saved next
root: it must remain replicated.
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
Its outer axis is `META_POP_AXIS_NAME` (`"META_POP"`); evolutionary algorithms
keep `POP_AXIS_NAME` (`"POP"`) for their own populations. Rollout and other VMA-only
casts read the active Manual axes from `jax.sharding.get_abstract_mesh()` at trace
time, including under `jit` and `vmap`. They do not require a Workflow field or
an axis argument in the rollout API, and their forward casts do not communicate.
Keep loss and statistics reduction axes explicit: population-parallel
execution does not imply that independent policies should synchronize updates.
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
mean there is no communication. DP averages the scalar loss with `pmean` before
differentiation; autodiff generates the corresponding gradient synchronization.
Metrics/running statistics are also reduced. EC and shared-buffer PBT gather data.
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
Shared-policy updates use `pmean_loss_and_grad`: the primary scalar loss is
averaged over the explicit DP axis inside `value_and_grad`, yielding the global
mean loss and gradients without a second gradient reduction. Auxiliary outputs
remain local and unchanged, including replay priorities; only their logged
metrics are reduced afterward. With `axis_name=None`, the helper performs
ordinary local differentiation, including for independent PBT members. Tests
check gradient values against a global-batch reference and preserve differing
per-device auxiliary values. They also compare actual replicas and inspect compiled
collectives: PPO/SAC use all-reduce, and a local-only computation has none.
The workflow tests cover distributed initialization and PBT shard-map paths,
plus RL algorithms, OpenES and RandomAgent setup/step/evaluate on two CPU devices.
Uniform, prioritized and LAP replay tests cover unequal masked insertion, empty
writes, wraparound, local sampling and priority updates against independent local
references. Their compiled buffer steps must contain no collective communication.
The supported JAX versions run these checks with and without outer JIT.
See [tests/README.md](../../tests/README.md) for the test layout, selection commands,
and CPU device configuration. Distributed tests run directly under pytest;
the two-axis VMA check uses a separate `--cpu-devices=4` invocation.
RL's `enable_jit` compiles setup as well as step and evaluation, so
initialization uses the same compiled execution mode.


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
   for `.github/workflows/release.yml` before publishing a GitHub Release.

### Tag the release commit

A tag identifies the commit to release. Creating or pushing a tag does not
publish packages or create a GitHub Release:

```shell
git switch main
git status --short  # Release changes must already be committed.
git tag -a v0.1.2 -m 'Version 0.1.2'
git push origin v0.1.2
```

Replace the example version before running. The tag must match
`evorl.__version__`: `v0.1.2` requires `0.1.2`. Release candidates use a version
such as `0.2.0rc1` and tag `v0.2.0rc1`, marked as a GitHub prerelease.

Annotated and lightweight tags are both supported; tag messages are not used
as release notes.

### Publish a GitHub Release

Open **Releases → Draft a new release**, select the version tag, and write the
release title and description. Focus on user-visible changes and compatibility.
Click **Publish release** when ready. Saving a draft does not start publishing.

Alternatively, write the description in a local Markdown file and publish with
GitHub CLI:

```shell
gh release create v0.1.2 --verify-tag --title 'v0.1.2' --notes-file release-notes.md
```

Add `--prerelease` for a release candidate. Publishing either a stable release or
a prerelease triggers the release workflow. It resolves the release's tag to an
exact commit, builds and tests that commit through CI, then uploads the same wheel
and sdist to PyPI and the existing GitHub Release in parallel. The GitHub Release
is already public while these jobs run; check that the workflow succeeds before
announcing package availability.

The workflow does not create releases or generate or change their title and
description. To refine the description later, use **Releases → version → Edit**;
editing it does not trigger another package upload. GitHub Release text does not
update PyPI metadata; PyPI displays the packaged README and a **Changelog** link
to GitHub Releases.

### Handle a failed release

If a publishing job fails, correct its configuration and rerun only the failed
jobs. The PyPI and GitHub Release jobs run independently: retrying a failed PyPI
job does not upload GitHub assets again, and retrying a failed GitHub job does not
upload the package to PyPI again. The GitHub job updates distribution assets from
the current run. If any PyPI files have already uploaded, check the published files
before retrying; an upload can fail after partial success.

PyPI files cannot be replaced for an already uploaded version. If a release needs
code changes, update the version, create a new tag, and publish a new GitHub Release.
Do not move or reuse a published version tag. Editing the GitHub Release
description is safe and does not change the uploaded package.
