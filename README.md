# EvoRL

EvoRL is an efficient Evolutionary Reinforcement Learning Framework integrated with [EvoX](https://github.com/EMI-Group/evox) (A JAX-accelerated EC Framework)

## Setup

For developer: see [CONTRIBUTING.md](./CONTRIBUTING.md)

For normal user:
```
pip install -e .
```

## Usage

EvoRL uses [hydra](https://hydra.cc/) to manage configs and run algorithms.

### Train agents from cli:

```shell
python -m evorl.train agent=exp/ppo/brax/ant env=brax/ant

python -m evorl.train agent=exp/ppo/brax/ant env=gymnax/CartPole-v1 agent_network.continuous_action=false

# sweep over multiple config values (seed=114 or seed=514):
python -m evorl.train -m agent=exp/ppo/brax/ant env=brax/ant seed=114,514
```

### Train multiple agents in parallel:

```shell
# need to install joblib plugin before the first run
pip install -U hydra-joblib-launcher

# sweep over multiple config values in parallel (for multi-GPU case)
python -m evorl.train_dist -m agent=exp/ppo/brax/ant env=brax/ant seed=114,514 hydra/launcher=joblib

# sweep over multiple config values in sequence
python -m evorl.train_dist -m agent=exp/ppo/brax/ant env=brax/ant seed=114,514

# optional: specify the gpu ids used for parallel training
CUDA_VISIBLE_DEVICES=0,5 python -m evorl.train_dist -m agent=exp/ppo/brax/ant env=brax/ant seed=114,514 hydra/launcher=joblib

```

Note:

- It's recommended to run every job on a single device. By default, the script will use all detected GPUs and run every job on a dedicated GPU.

- If you persist in parallel training on a single device, set environment variables like `XLA_PYTHON_CLIENT_MEM_FRACTION=.10` or `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid the OOM from JAX's pre-allocation.

- If the number of submitted jobs exceeds the number of CPU cores, joblib will wait and reuse previous processes. This could cause misconfigured GPU settings. To solve it, append `hydra.launcher.n_jobs=<#jobs>` to the script.

## Acknowledgement

- brax
- acme
- evox
- gymnax
- jumanji
- jaxmarl
