# Installation



## Setup

EvoRL is based on `jax`. So `jax` should be installed first, please follow [JAX official installation guide](https://jax.readthedocs.io/en/latest/quickstart.html#installation).

Then install EvoRL from source:

```shell
# Install the evorl package from source
git clone https://github.com/EMI-Group/evorl.git
cd evorl
pip install -e .
```

## RL Environments

By default, `pip install -e .` will automatically install environments on `brax`. If you want to install other supported environments, you need manually install the related environment packages. We provide useful extras for different environments.

```shell
# ===== GPU-accelerated Environments =====
# Mujoco playground Envs:
pip install -e ".[mujoco-playground]"
# gymnax Envs:
pip install -e ".[gymnax]"
# Jumanji Envs:
pip install -e ".[jumanji]"
# JaxMARL Envs:
pip install -e ".[jaxmarl]"

# ===== CPU-based Environments =====
# EnvPool Envs: (also require py<3.12)
pip install -e ".[envpool]"
# Gymnasium Envs:
pip install -e ".[gymnasium]"
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
