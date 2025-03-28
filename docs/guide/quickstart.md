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
# except the wandb's group name is different.
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
```

For [multi-run mode](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/), `scripts/train_dist.py` has the similar behavior as `scripts/train.py` when lauching without `joblib`. The only difference is that `scripts/train_dist.py` will set the group name of WandB by the experiment name, while `scripts/train.py` uses `"dev"` as the group name for all runs.

However, when there are multiple GPUs, `scripts/train.py` will sequentially run a single run across multiple GPUs if the related algorithm supports distributed training. Instead, if enabling `joblib`, `scripts/train_dist.py` will run multiple jobs in parallel, where each job is running on a single GPU. If there are more jobs than #GPUs, multiple jobs will parallelly execute on the same GPU. For example, if there are 3 jobs to be executed on 2 GPUs, GPU1 will run job1 and job3, while GPU2 will run job2.

:::{admonition} Tips for `scripts/train_dist.py`
:class: tip

- It only supports multi-run mode, i.e, using `python scripts/train_dist.py -m` to launch the training, even if there is only one config to run.
- It's recommended to run every job on a single device. By default, the script will use all detected GPUs and run every job on a dedicated GPU.
  - If you want to run mulitple jobs on a single device, set environment variables like `XLA_PYTHON_CLIENT_MEM_FRACTION=.10` or `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid the OOM due to the JAX's pre-allocation.
- If the number of submitted jobs exceeds the number of CPU cores, `joblib` will wait and reuse previous processes.  This is a caveat of `joblib` and could cause misconfigured GPU settings. However, this is a rare case. To solve it, append `hydra.launcher.n_jobs=<#jobs>` to the script.
- Unlike `scripts/train.py`, this module only supports Nvidia GPUs.
:::

### Logging

When not using multi-run mode (without `-m`), the outputs will be stored in `./outputs`. When using multi-run mode (`-m`), the outputs will be stored in `./multirun`. Specifically, when launching algorithms from the training scripts, the log file and checkpoint files will be stored in `./outputs|multirun/train|train_dist/<timestamp>/<exp-name>/`.

By default, the script will enable two recorders for logging: `LogRecorder` and `WandbRecorder`. `LogRecorder` will save logs (`*.log`) in the above path, and `WandbRecorder` will upload the data to [WandB](https://wandb.ai/site/), which provides beautiful visualizations.

````{tip}
To disable the WandB logging or use its offline mode, set environment variable `WANDB_MODE` before launching the training:

```shell
WANDB_MODE=disabled python scripts/train.py agent=ppo env=brax/ant
WANDB_MODE=offline python scripts/train.py agent=ppo env=brax/ant
```
````


## Custom Training under Python API

Besides training from CLI, you can also start the training through the following python codes:

```{include} ../_static/train_demo.py
:literal:
:language: python
```
