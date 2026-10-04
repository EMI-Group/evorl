# Installation



## Setup

EvoRL is based on `jax`. So `jax` should be installed first, please follow [JAX official installation guide](https://jax.readthedocs.io/en/latest/quickstart.html#installation).

Install the released package from PyPI (available after the first release):

```shell
pip install evorl-jax
```

The PyPI distribution is named `evorl-jax`, while Python code uses `import evorl`.
For development or the CLI training scripts and configs, install from source:

```shell
# Install the evorl package from source
git clone https://github.com/EMI-Group/evorl.git
cd evorl
pip install -e .
```

## Experiment Logging

Aim is the default experiment tracker and is installed by `pip install evorl-jax`
(or `pip install -e .` from source). The other tracking SDKs are optional. Install
an EvoRL extra or install the SDK directly into the same Python environment:

| Recorder | EvoRL extra (from PyPI) | Direct SDK installation |
| --- | --- | --- |
| Aim | Included in `pip install evorl-jax` | `pip install aim` |
| WandB | `pip install "evorl-jax[wandb]"` | `pip install wandb` |
| SwanLab | `pip install "evorl-jax[swanlab]"` | `pip install swanlab` |
| Comet | `pip install "evorl-jax[comet]"` | `pip install comet_ml` |
| Neptune | `pip install "evorl-jax[neptune]"` | `pip install neptune-scale` |

For a source checkout, use `pip install -e ".[extra]"` instead.
Extras can be combined with each other and with environment extras:

```shell
pip install "evorl-jax[wandb,swanlab,gymnax]"
```

Installing an SDK does not enable its recorder. Select the installed backends
with the `recorders` override; quote the list to prevent shell globbing:

```shell
# Default local logging and Aim; view runs using: aim up --repo aim
python scripts/train.py agent=ppo env=brax/ant

# Use WandB instead of Aim (authenticate using wandb login for online logging)
python scripts/train.py agent=ppo env=brax/ant 'recorders=[log,wandb]'

# Record to multiple installed backends
python scripts/train.py agent=ppo env=brax/ant 'recorders=[log,aim,swanlab]'
```

Only enabled tracking SDKs are imported, when their recorder initializes.
Cloud backends also require their own credentials and a valid project. Neptune's
hosted service was discontinued on March 5, 2026; its extra is retained for
compatibility with functioning endpoints. See [Logging in the quickstart](quickstart.md#logging)
for backend mappings, grouping, and customization.

## RL Environments

By default, `pip install evorl-jax` (or `pip install -e .` from source) will automatically install environments on `brax`. If you want to install other supported environments, you need manually install the related environment packages. We provide useful extras for different environments. For a source checkout, use `pip install -e ".[extra]"` instead.

```shell
# ===== GPU-accelerated Environments =====
# Mujoco playground Envs:
pip install "evorl-jax[mujoco-playground]"
# gymnax Envs:
pip install "evorl-jax[gymnax]"
# Jumanji Envs:
pip install "evorl-jax[jumanji]"
# JaxMARL Envs:
pip install "evorl-jax[jaxmarl]"

# ===== CPU-based Environments =====
# EnvPool Envs:
pip install "evorl-jax[envpool]"
# Gymnasium Envs:
pip install "evorl-jax[gymnasium]"
```

| Environment Library                                                        | Descriptions                            |
| -------------------------------------------------------------------------- | --------------------------------------- |
| [Brax](https://github.com/google/brax)                                     | Robotic control                         |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)  | Robotic control                         |
| [gymnax (experimental)](https://github.com/RobertTLange/gymnax)            | classic control, bsuite, MinAtar        |
| [JaxMARL (experimental)](https://github.com/FLAIROx/JaxMARL)               | Multi-agent Envs                        |
| [Jumanji (experimental)](https://github.com/instadeepai/jumanji)           | Game, Combinatorial optimization        |
| [EnvPool (experimental)](https://github.com/sail-sg/envpool)               | High-performance CPU-based environments |
| [Gymnasium (experimental)](https://github.com/Farama-Foundation/Gymnasium) | Standard CPU-based environments         |

```{attention}
These experimental environments have limited supports, some algorithms are incompatible with them.
```

```{attention}
Users with NVIDIA Ampere architecture GPUs (e.g., RTX 30 and 40 series) may experience reproducibility issues in `mujoco_playground` due to JAX’s default use of TF32 for matrix multiplications. See [Reproducibility / GPU Precision Issues](https://github.com/google-deepmind/mujoco_playground?tab=readme-ov-file#reproducibility--gpu-precision-issues)
```

For CPU-based Envs, please refer to the following API References:

- EnvPool: [`evorl.envs.envpool`](#evorl.envs.envpool)
  - Use C++ Thread Pool, more efficient than Gymnasium.
- Gymnasium: [`evorl.envs.gymnasium`](#evorl.envs.gymnasium)
  - Use Python `multiprocessing`. The most commonly used Env API.
