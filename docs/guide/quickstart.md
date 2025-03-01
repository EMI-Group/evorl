# Quickstart

This document provides a quick overview about how to use EvoRL to train different algorithms.

## Training

EvoRL uses [hydra](https://hydra.cc/) to manage configs and run algorithms. We provide a module `evorl.train` to run algorithms from CLI. It follows [hydra's CLI syntax](https://hydra.cc/docs/advanced/hydra-command-line-flags/).

```shell
python -m evorl.train agent=ppo env=brax/ant

# override some config by hydra cli override syntax:
# https://hydra.cc/docs/advanced/override_grammar/basic/
python -m evorl.train agent=ppo env=brax/ant seed=42 discount=0.995 \
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

### Advanced usage

Module `evorl.train` also supports hydra's [multi-run mode](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/). For example, if you want to perform 5 runs with different seeds, you can use the following command:

```shell
# specify the seed manually
python -m evorl.train -m agent=exp/ppo/brax/ant env=brax/ant seed=0,1,2,3,4
# or use hydra's extended override syntax:
python -m evorl.train -m agent=exp/ppo/brax/ant env=brax/ant seed=range(5)
```

Similarly, it allows seeping the config options with hydra's [extended override syntax](https://hydra.cc/docs/advanced/override_grammar/extended/). It is easy to perform a hyperparameter grid search:

```shell
python -m evorl.train -m agent=exp/ppo/brax/ant env=brax/ant gae_lambda=range(0.8,0.95,0.01) discount=0.99,0.999,0.9999
```

However, `evorl.train` is used to running experiments sequentially. To support massive number of experiments in parallel, we also provide the module `evorl.train_dist` to run experiments synchronously across different GPUs.

For [multi-run mode](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/), `evorl.train_dist` has the similar behavior as `evorl.train` when there is a single GPU. The only difference is that `evorl.train_dist` will set the group name of WandB by the experiment name, while `evorl.train` uses `"dev"` as the group name for all runs.

When there are multiple GPUs, `evorl.train` will sequentially run a single run across multiple GPUs if the related algorithm supports multi-GPU training. Instead, `evorl.train_dist` will run multiple runs in parallel, where each training instance is running on a single dedicated GPU. Therefore, for `evorl.train_dist`, the number of runs should not be higher than the number of GPUs.

For single GPU case:

```shell
# this is similar to evorl.train for a single GPU case, except the wandb's group name is different.
python -m evorl.train_dist -m agent=exp/ppo/brax/ant env=brax/ant seed=114,514
```

For multiple GPUs case:

```shell
# sweep over multiple config values in parallel (using multi-process)
python -m evorl.train_dist -m agent=exp/ppo/brax/ant env=brax/ant seed=114,514 hydra/launcher=joblib

# optional: specify the gpu ids used for parallel training
CUDA_VISIBLE_DEVICES=0,5 python -m evorl.train_dist -m agent=exp/ppo/brax/ant env=brax/ant seed=114,514 hydra/launcher=joblib
```

:::{admonition} Tips for `evorl.train_dist`
:class: tip

- It's recommended to run every job on a single device. By default, the script will use all detected GPUs and run every job on a dedicated GPU.
  - If you persist in parallel training on a single device, set environment variables like `XLA_PYTHON_CLIENT_MEM_FRACTION=.10` or `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid the OOM from the JAX's pre-allocation.
- If the number of submitted jobs exceeds the number of CPU cores, `joblib` will wait and reuse previous processes. This is a caveat of `joblib` and could cause misconfigured GPU settings. To solve it, append `hydra.launcher.n_jobs=<#jobs>` to the script.
- Unlike `evorl.train`, this module is written for Nvidia GPUs.
:::

### Logging

When not using multi-run mode (without `-m`), the outputs will be stored in `./outputs`. When using multi-run mode (`-m`), the outputs will be stored in `./multirun`. Specifically, when launching algorithms from the training scripts, the log file and checkpoint files will be stored in `./outputs|multirun/train|train_dist/<timestamp>/<exp-name>/`.

By default, the script will enable two recorders for logging: `LogRecorder` and `WandbRecorder`. `LogRecorder` will save logs (`*.log`) in the above path, and `WandbRecorder` will upload the data to [WandB](https://wandb.ai/site/), which provides beautiful visualizations.

````{tip}
To disable the WandB logging or use its offline mode, set environment variable `WANDB_MODE` before launching the training:

```shell
WANDB_MODE=disabled python -m evorl.train agent=ppo env=brax/ant
WANDB_MODE=offline python -m evorl.train agent=ppo env=brax/ant
```
````


## Custom Training under Python API

Besides training from CLI, you can also start the training through the python codes:

```{include} ../_static/train_demo.py
:literal:
:language: python
```
