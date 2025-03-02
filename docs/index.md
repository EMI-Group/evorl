# Welcome to EvoRL!

```{toctree}
:caption: Getting started
:maxdepth: 1
:hidden:

guide/installation
guide/quickstart
guide/key-concepts
```

```{toctree}
:caption: Tutorials
:maxdepth: 1
:hidden:

tutorial/algorithms
```

```{toctree}
:caption: 'References'
:maxdepth: 1
:hidden:

apidocs/index
```

```{toctree}
:caption: 'Developer Guide'
:maxdepth: 1
:hidden:

dev/contributing
```

EvoRL is a fully GPU-accelerated framework for Evolutionary Reinforcement Learning, implemented with JAX.

## Key Features

- **End-to-end training pipelines**: The training pipelines for RL, EC and EvoRL are executed on GPUs, eliminating dense communication between CPUs and GPUs in traditional implementations and fully utilizing the parallel computing capabilities of modern GPU architectures. Besides, most algorithms has a `Workflow.step()` function that is capable of `jax.jit` and `jax.vmap()`, supporting parallel training and JIT on full computation graph. The maximum seed-up is up to 60x depend on the algorithms.
- **Easy integration between EC to RL**: Due to modular design, EC components can be easily plug-and-play in workflows and cooperate with RL.
- **Unified Environment API**: Support multiple GPU-accelerated RL environment packages (eg: Brax, gymnax, ...) under the same Environment API. Besides, multiple useful EnvWrappers are provided.
