# Quickstart

This document provides a quick overview about how to use EvoRL to train different algorithms.

## Training

EvoRL uses [hydra](https://hydra.cc/) to manage configs and run algorithms. We provide a script `scripts/train.py` to run algorithms from CLI. It follows [hydra's CLI syntax](https://hydra.cc/docs/advanced/hydra-command-line-flags/).

```shell
python scripts/train.py agent=ppo env=brax/ant

# override some configs
python scripts/train.py agent=ppo env=brax/ant seed=42 discount=0.995 \
    agent_network.actor_hidden_layer_sizes="[128,128]"
```

### Configs

Hydra uses a modularized config file structures. Config files are some `*.yaml` files in the directory `configs/` with the following hierarchy:

```text
# hierarchy of folder `configs/`
configs
├── agent
│   ├── ppo.yaml
│   ├── ...
...
├── config.yaml
├── env
│   ├── brax
│   │   ├── ant.yaml
│   │   ├── ...
│   ├── envpool
│   └── gymnax
└── logging.yaml
```

- `configs/config.yaml` is the top-level config template, which imports other config files as its components.
- `configs/agent` defines the configs for algorithms.
  - Specifically, `configs/agent/exp` defines the algorithm configs we tuned for experiments.
- `configs/env` defines the configs for environments.

We list some common fields in the final config, which is useful as options passing into the above training script:

- `agent`: Specify the algorithm's config file. The `.yaml` suffix is not needed.
- `env`: Specify the environment's config file. The `.yaml` suffix is not needed.
- `seed`: Random seed.
- `checkpoint.enable`: Whether to save the checkpoint files during training. Default is `false`.
- `enable_jit`: Whether to enable JIT compilation for the workflow.
- `recorders`: Logging backends. Defaults to `[log, aim]`; install optional SDKs before selecting other backends.

Moreover, to use the own config folders instead of the EvoRL's `configs/`, you can specify the config path by the `-cp` and `-cn` option. This is helpful when you want to use EvoRL as a package and build your own project. More details can be found in the [hydra's CLI syntax](https://hydra.cc/docs/advanced/hydra-command-line-flags/).

```shell
python scripts/train.py -cp /path/to/your/configs -cn /path/to/your/configs/your_global_cfg.yaml agent=ppo env=brax/ant
```

### Advanced usage

Script `scripts/train.py` also supports hydra's [multi-run mode](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/). For example, if you want to perform 5 runs with different seeds, you can use the following command:

```shell
# specify the seed manually
python scripts/train.py -m agent=exp/ppo/brax/ant env=brax/ant seed=0,1,2,3,4
# or use hydra's extended override syntax:
python scripts/train.py -m agent=exp/ppo/brax/ant env=brax/ant seed=range(5)
```

Similarly, it allows seeping the configs with hydra's [extended override syntax](https://hydra.cc/docs/advanced/override_grammar/extended/). It is easy to perform a hyperparameter grid search, for example:

```shell
python scripts/train.py -m agent=exp/ppo/brax/ant env=brax/ant \
    gae_lambda=range(0.8,0.95,0.01) discount=0.99,0.999,0.9999
```

However, `scripts/train.py` is used to run experiments sequentially. To support massive number of experiments in parallel, we also provide the module `scripts/train_dist.py` to run multiple experiments synchronously across different GPUs. Below are some examples:

For single GPU case:

```shell
# this is similar to train.py for a single GPU case,
# except the recorder group is different.
python scripts/train_dist.py -m agent=exp/ppo/brax/ant env=brax/ant seed=114,514
```

For multiple GPUs case:

```shell
# sweep over multiple config values in parallel (using multi-process)
python scripts/train_dist.py -m hydra/launcher=joblib \
    agent=exp/ppo/brax/ant env=brax/ant seed=114,514

# optional: specify the gpu ids used for parallel training
CUDA_VISIBLE_DEVICES=0,5 python scripts/train_dist.py -m hydra/launcher=joblib \
    agent=exp/ppo/brax/ant env=brax/ant seed=114,514

# AMD GPUs: use the ROCm runtime's visibility mask
ROCR_VISIBLE_DEVICES=0,5 python scripts/train_dist.py -m hydra/launcher=joblib \
    agent=exp/ppo/brax/ant env=brax/ant seed=114,514
```

For [multi-run mode](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/), `scripts/train_dist.py` has the similar behavior as `scripts/train.py` when launching without `joblib`. It uses the experiment name as the recorder group, while `scripts/train.py` uses `"dev"`. The shared script helper computes these groups from the entrypoint's training mode.

However, when there are multiple GPUs, `scripts/train.py` will sequentially run a single run across multiple GPUs if the related algorithm supports distributed training. Instead, if enabling `joblib`, `scripts/train_dist.py` will run multiple jobs in parallel, where each job is running on a single GPU. If there are more jobs than #GPUs, multiple jobs will parallelly execute on the same GPU. For example, if there are 3 jobs to be executed on 2 GPUs, GPU1 will run job1 and job3, while GPU2 will run job2.

:::{admonition} Tips for `scripts/train_dist.py`
:class: tip

- It only supports multi-run mode, i.e, using `python scripts/train_dist.py -m` to launch the training, even if there is only one config to run.
- It's recommended to run every job on a single device. By default, the script will use all detected GPUs and run every job on a dedicated GPU.
  - If you want to run mulitple jobs on a single device, set environment variables like `XLA_PYTHON_CLIENT_MEM_FRACTION=.10` or `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid the OOM due to the JAX's pre-allocation.
- GPU discovery runs in a short subprocess using JAX's default backend. The script supports NVIDIA/CUDA and AMD/ROCm with the corresponding JAX GPU installation. CPU and TPU backends are rejected explicitly.
- The selected backend's registered name (`cuda` or `rocm`) determines whether the script sets `JAX_CUDA_VISIBLE_DEVICES` or `JAX_ROCM_VISIBLE_DEVICES` before importing JAX. Existing runtime masks (`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `HIP_VISIBLE_DEVICES`) are preserved.
- Discovery initializes JAX only in the probe subprocess and adds startup overhead. Training continues in the existing Hydra/Joblib process; this change does not fix device rebinding after JAX has initialized in a reused process.
:::

### Logging

With the default Hydra configuration, a single run stores outputs in `outputs/<script>/<timestamp>/`, and multi-run mode (`-m`) uses `multirun/<script>/<timestamp>/<overrides>/`, where `<script>` is `train` or `train_dist`. `LogRecorder` writes `<experiment-name>.log` there; checkpoints use the `checkpoints/` subdirectory when `checkpoint.enable=true`. Aim uses the separate shared repository described below.

By default, the script enables `LogRecorder` and `AimRecorder`. `LogRecorder` saves `*.log` in the output directory. Aim stores runs in the shared `aim/.aim` repository under the original working directory, so separate training runs can be compared with `aim up --repo aim`. Aim is installed with EvoRL; the other tracking SDKs are optional.

Install optional trackers using an EvoRL extra or their SDK package directly:

```shell
# From the EvoRL repository root
pip install -e ".[wandb,swanlab]"
# Alternatively, install only the SDKs into your existing EvoRL environment
pip install wandb swanlab

python scripts/train.py agent=ppo env=brax/ant 'recorders=[log,wandb,swanlab]'
```

See [Experiment Logging installation](installation.md#experiment-logging) for all backend extras and SDK package names. Installing a backend does not select it automatically; `recorders` controls which trackers are enabled.

The `recorders` list can contain `log`, `aim`, `wandb`, `swanlab`, `comet`, or `neptune`. Backend SDKs are imported when their recorder starts; importing EvoRL does not require them. Each backend receives the same `project`, run `name`, resolved `config`, `tags`, `path`, and `group` values. `scripts/recorder_setup.py` builds the name, tags, paths, and group in one place. `train.py` calls `setup_recorders(config, workflow_cls.name())`; `train_dist.py` adds `parallel=True`. This flag selects the entrypoint's grouping convention, independently of GPU count or whether Joblib is enabled.

To customize a backend, add `recorder_kwargs` to that script's call:

```python
recorders = setup_recorders(
    config,
    workflow_cls.name(),
    recorder_kwargs={"aim": {"path": "/shared/aim-repo", "group": "ablation"}},
)
```

Each tracking backend's dictionary can override common constructor values or supply SDK-specific initialization options. `recorder_kwargs` accepts `aim`, `wandb`, `swanlab`, `comet`, and `neptune`; misspelled backend keys raise an error. `LogRecorder` uses the script's log path and console defaults. Keep `parallel=True` when customizing the `train_dist.py` call. Grouping and backend options remain script conventions and are not fields in `logging.yaml`; no separate `group` argument is required by `setup_recorders`.

The common constructor parameters are EvoRL concepts, not identical SDK features:

| Recorder | `project` and `name` | `group` | `path` |
| --- | --- | --- | --- |
| WandB | Native project and run name | Native group | Local run directory |
| Aim | Project metadata and native run name | Experiment | Shared Aim repository |
| SwanLab | Native project and experiment name | Native group | Local log directory |
| Comet | Native project and experiment name | `group` metadata, selectable for UI grouping | `output_dir` metadata only |
| Neptune | Native `workspace/project` and experiment name | Group tag | Local log directory |

All tracking recorders receive the resolved config and tags, but config appears in each SDK's own config, parameter, or metadata view. SDK-specific options are not portable across backends. Nested metric dictionaries are flattened to slash-separated names outside WandB. `Series` values become distributions or histograms; SwanLab uses an ECharts bar chart. Aim, SwanLab, and Neptune use 64 bins, matching WandB's default; Comet uses its SDK's histogram binning. Empty or nonfinite histogram samples and `None` values are skipped outside WandB. Raw arrays and other nonnumeric values are retained as JSON text outside WandB, so their dashboards differ from WandB's native array logging. SDKs may also capture system data or console output according to their own defaults; EvoRL does not normalize this automatic logging.

In this project, `DataFrame` logging records the PBT population's hyperparameters for an iteration: each row is a population slot identified by `pop_id`, and the other columns are the searched hyperparameters. It records the population used for that iteration's metrics, before exploitation and exploration. The recorder does not add a step column or mutate the frame.

| Recorder | Table representation | Step association |
| --- | --- | --- |
| WandB | Native `Table` | Enclosing `Run.log(..., step=step)` history entry |
| Aim | JSON `Text` snapshot | `Run.track(..., step=step)` |
| SwanLab | ECharts table | Enclosing `Run.log(..., step=step)` |
| Comet | JSON table asset | `Experiment.set_step(step)` before `log_table`; the asset inherits that step |
| Neptune | JSON string series | `Run.log_string_series(..., step=step)` |

Comet requires an asset filename: the recorder uses the encoded metric key with a `.json` suffix. Snapshots retain their asset step and are not overwritten. The adapter does not create a local table file. Aim and Neptune preserve each snapshot and its step, but offer no native table view through these APIs. The neptune-scale SDK limits each string-series data point to 1 MiB; larger table snapshots would require a file-series adapter.

Every EvoRL recorder requires `write(data, step: int)`. Omitted steps, `None`, booleans, and noninteger values are rejected. NumPy and JAX integer scalars are converted to Python integers. Existing workflows already pass their iteration explicitly, and multiple writes in an iteration use the same step. Comet stores a current step internally: the adapter sets it once at the start of each write, then all metrics, histograms, text, and tables inherit it. Neptune's logging APIs require a step on each call. EvoRL always uses the supplied iteration and never generates an internal counter.

WandB marks an interrupted run as failed on close, and SwanLab marks it as crashed. Aim, Comet, and Neptune only close or finalize their runs and do not currently set an equivalent failure state.

Neptune's hosted service was discontinued on March 5, 2026. The `neptune` recorder is a compatibility adapter for installations with a functioning Neptune endpoint; it is not a usable default cloud backend. Neptune requires a `workspace/project` project identifier, which can be supplied by the script with `recorder_kwargs={"neptune": {"project": "workspace/project"}}`.

````{tip}
When selecting WandB, set `WANDB_MODE` to disable its logging or use offline mode:

```shell
WANDB_MODE=disabled python scripts/train.py agent=ppo env=brax/ant 'recorders=[log,wandb]'
WANDB_MODE=offline python scripts/train.py agent=ppo env=brax/ant 'recorders=[log,wandb]'
```
````


## Custom Training under Python API

The default `recorders: [log, aim]` selection is applied by the training scripts.
When using the Python API directly, a workflow starts with no recorders; add them
explicitly with `workflow.add_recorders(...)`. The example below adds only
`LogRecorder`.

Besides training from CLI, you can also start the training through the following python codes:

```{include} ../_static/train_demo.py
:literal:
:language: python
```
