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
pip install "evorl-jax[wandb,swanlab,jumanji]"
```

Installing an SDK does not enable its recorder. Select the installed backends
with the `recorders` override; quote the list to prevent shell globbing:

```shell
# Default local logging and Aim; view runs using: aim up --repo <output_dir>
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

## Optional EvoX algorithms

```shell
uv pip install "evorl-jax[evox]"
# From source:
uv pip install -e ".[evox]"
```

Requires EvoX `>=1.4,<2` and PyTorch `>=2.9`. Choose a PyTorch build supporting
the accelerator used by JAX. See the [CSO and RVEA examples](../tutorial/algorithms.md#optional-pytorch-evox-integration)
for usage and execution constraints.

## RL Environments

Brax and MuJoCo Playground are supported by default and installed with
`pip install evorl-jax` (or `pip install -e .` from source). Install extras
for other environment libraries. For a source checkout, use
`pip install -e ".[extra]"` instead.

```shell
# ===== GPU-accelerated Environments =====
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

### Multi-objective rewards

Single-agent rewards are scalar arrays with shape `()`. Brax and MuJoCo
Playground configurations can set `reward_keys` to select scalar per-step
fields from `info.metrics` as a reward vector. Key order defines objective
order; even one selected key produces shape `(1,)`. Omitting `reward_keys`
preserves the native scalar reward.

The selected metrics retain their native signs, weights and time scaling;
they need not sum to the native total reward. Episode returns preserve the
objective axis, while done flags and episode lengths remain per environment.
Multi-agent rewards remain agent-keyed mappings.

Examples are `brax/ant_multi_objective` and
`playground/dm_control_suite/CartpoleBalance_multi_objective`.

### Jumanji

Jumanji single-agent tasks use `Env`; multi-agent tasks use `MultiAgentEnv` with
agent-keyed actions, observations, rewards and done flags. Individual rewards
are preserved; native shared rewards are copied unchanged to each agent and
retained under `reward["__all__"]`. `done["__all__"]` marks the episode boundary.

Structured observations use nested dictionaries and `SpaceContainer`, preserving
native dtypes and action masks. Multi-agent observations separate agent-specific
fields from shared fields and include an `agent_id`; `info.global_obs` retains
the native joint observation. Masks are provided to policies without enforcing
action selection. Single-agent MultiDiscrete actions remain integer arrays.
SearchAndRescue's continuous actions retain their native `[-1, 1]` bounds.

Set `max_episode_steps` as the shared episode limit. Tasks with a native
`time_limit` receive it at construction, and EvoRL's episode wrapper uses the
same value. Wrapped environments preserve native termination and truncation; tasks
without a native time limit are externally truncated. Native rewards and
generators are unchanged. Autoreset uses EvoRL's wrappers.
YAML configurations are in `configs/env/jumanji/`, grouped
by task family. Default Sokoban requires an external level dataset and has no
bundled YAML.

### Gymnax (best effort)

Gymnax 1.0.1 requires JAX below 0.12 and Gymnasium below 1.4. No dependency
overrides are needed. From a source checkout, create an environment and install
compatible versions together:

```shell
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e ".[gymnax,jaxmarl]" "gymnax>=1.0.1" "jax[cuda13]>=0.10.0,<0.12"
```

The CUDA 13 extra enables NVIDIA GPU support. For CPU-only installation, replace
`jax[cuda13]` with `jax` while keeping the version bounds. See the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for
platform and driver requirements.

### JaxMARL (best effort)

JaxMARL depends on Gymnax; the command above installs both.

JaxMARL 0.2.0 or newer uses `MultiAgentEnv` with EvoRL episode and autoreset
wrappers. Configurations cover MPE, SMAX, Switch Riddle, Hanabi, OvercookedV2,
CoinGame and JaxNav, including continuous and discrete JaxNav actions. Set native
constructor options such as `layout` and `act_type` at the top level of the
environment YAML. Explicit arguments to `create_env()` override YAML values.

SMAX and Hanabi observations contain `observation` and `action_mask` fields.
Native global observations are available in `info.global_obs` when provided.
MABrax has been removed. STORM's nonuniform native APIs and the separately
installed JaxRobotarium environments are not included in this adapter.

### Environment support

| Environment Library                                                    | Description                            |
| ---------------------------------------------------------------------- | -------------------------------------- |
| [Brax](https://github.com/google/brax)                                   | Robotic control                        |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) | Robotic control                        |
| [Jumanji](https://github.com/instadeepai/jumanji)                         | Games and combinatorial optimization   |
| [EnvPool](https://github.com/sail-sg/envpool)                             | High-performance CPU-based environments |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)               | Standard CPU-based environments        |
| [Gymnax (best effort)](https://github.com/RobertTLange/gymnax)             | Classic control, bsuite and MinAtar     |
| [JaxMARL (best effort)](https://github.com/FLAIROx/JaxMARL)                | Multi-agent environments               |

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
