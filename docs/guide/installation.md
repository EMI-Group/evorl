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

By default, `pip install evorl` will automatically install environments on `brax`. If you want to install other supported environments, you need manually install the related environment packages. For example:

```shell
# ===== GPU-accelerated Environments =====
# gymnax Envs:
pip install gymnax
# Jumanji Envs:
pip install jumanji
# JaxMARL Envs:
pip install jaxmarl

# ===== CPU-based Environments =====
# EnvPool Envs: (also require py<3.12)
pip install envpool "numpy<2.0.0"
# Gymnasium Envs:
pip install "gymnasium[atari,mujoco,classic-control,box2d]>=1.1.0"
```

| Environment Library                                                        | Descriptions                            |
| -------------------------------------------------------------------------- | --------------------------------------- |
| [Brax](https://github.com/google/brax)                                     | Robotic control                         |
| [gymnax (experimental)](https://github.com/RobertTLange/gymnax)            | classic control, bsuite, MinAtar        |
| [JaxMARL (experimental)](https://github.com/FLAIROx/JaxMARL)               | Multi-agent Envs                        |
| [Jumanji (experimental)](https://github.com/instadeepai/jumanji)           | Game, Combinatorial optimization        |
| [EnvPool (experimental)](https://github.com/sail-sg/envpool)               | High-performance CPU-based environments |
| [Gymnasium (experimental)](https://github.com/Farama-Foundation/Gymnasium) | Standard CPU-based environments         |

```{attention}
These experimental environments have limited supports, some algorithms are incompatible with them.
```

For CPU-based Envs, please refer to the following API References:

- EnvPool: [`evorl.envs.envpool`](#evorl.envs.envpool)
  - Use C++ Thread Pool, more efficient than Gymnasium.
- Gymnasium: [`evorl.envs.gymnasium`](#evorl.envs.gymnasium)
  - Use Python `multiprocessing`. The most commonly used Env API.
