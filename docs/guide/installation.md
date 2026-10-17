# Installation



## Setup

EvoRL requires Python 3.12 or newer and JAX/jaxlib 0.10.0 or newer. Install JAX first by following the [official installation guide](https://docs.jax.dev/en/latest/installation.html).

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
pip install "evorl-jax[wandb,swanlab,mujoco-playground]"
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

### Gymnax (best effort)

Gymnax 1.0.0 declares older JAX and Gymnasium version bounds. After installing
EvoRL, use uv to override these bounds without creating a separate file:

```shell
uv pip install "gymnax==1.0.0" --overrides - <<'EOF'
jax>=0.10.0
gymnasium>=1.4.0
EOF
```

### Environment support

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

Both CPU adapters support Box observations, zero-based Discrete observations and
nested Dicts of these spaces through `SpaceContainer`, with floating-point Box or
zero-based Discrete actions. Tuple spaces and multi-agent tasks require separate
adapters. EnvPool requires full synchronous
batches and next-step autoreset.
Observations retain their native layout and dtype. Agents must support the
task's observation encoding and action bounds; continuous policies assuming
`[-1, 1]` need `ActionSquashWrapper` for other native bounds.
