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

By default, `pip install evorl` will automatically install environments on `brax` and `gymnax`. If you want to install other supported environments, you need manually install the related environment packages. For example:

```shell
# EnvPool Envs:
pip install envpool "numpy<2.0.0"
# Jumanji Envs:
pip install jumanji
# JaxMARL Envs:
pip install jaxmarl
```

```{attention}
These experimental environments have limited supports, some algorithms are incompatible with them.
```
