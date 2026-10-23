# Algorithms

Currently, EvoRL supports various training pipelines (workflows):

1. Reinforcement Learning (RL) Algorithms
2. Evolutionary Computation (EC) Algorithms, specific for policy search
3. Evolutionary Reinforcement Learning (EvoRL):
    - Evolution-guided Reinforcement Learning (ERL)
    - Population-based AutoRL

This document introduces these types of algorithms implemented in EvoRL. All algorithms are defined in [`evorl.algorithms`](#evorl.algorithms).

Policy and value networks use Flax NNX. Workflow builders create agents directly
from YAML configuration and select observation preprocessing. Agents hold fixed
`agent_network_cfg` configuration and create networks in `init()` from the spaces
and random key.
`AgentState.model_state` stores the complete NNX `GraphState`, including
parameters and auxiliary variables. Training restores local models and
optimizers, updates the selected parameters, and returns new state snapshots.
Rollout and evaluation retain their explicit-state interfaces. Evolutionary
operators act only on `nnx.Param` variables.
Linen checkpoints are incompatible with the new parameter layout.

Agent loss methods take `AgentState` and return a loss dictionary. Workflows
differentiate working NNX models and pass a temporary state containing
`nnx.unpack(models)` to the loss. `Agent.get_models()` restores models sharing
those working variables, so forward-pass updates such as spectral normalization
are retained. Workflows create an isolated model copy before training and save
the updated snapshot afterward; rollout and evaluation use `copy=True`.
See [Agent state and NNX training](../dev/contributing.md#agent-state-and-nnx-training)
for the TD3 update sequence and variable-sharing contract.

## RL Algorithms

Supported RL Algorithms:

| Algorithm | Workflow                                                                    | Policy Type   | Supported Action Space |
| --------- | --------------------------------------------------------------------------- | ------------- | ---------------------- |
| Random    | [`RandomAgentWorkflow`](#evorl.algorithms.random_agent.RandomAgentWorkflow) | -             | Discrete & Continuous  |
| A2C       | [`A2CWorkflow`](#evorl.algorithms.a2c.A2CWorkflow)                          | Stochastic    | Discrete & Continuous  |
| PPO       | [`PPOWorkflow`](#evorl.algorithms.ppo.PPOWorkflow)                          | Stochastic    | Discrete & Continuous  |
| IMPALA    | [`IMPALAWorkflow`](#evorl.algorithms.impala.IMPALAWorkflow)                 | Stochastic    | Discrete & Continuous  |
| DQN       | [`DQNWorkflow`](#evorl.algorithms.dqn.DQNWorkflow)                          | Value-based   | Discrete               |
| DDPG      | [`DDPGWorkflow`](#evorl.algorithms.ddpg.DDPGWorkflow)                       | Deterministic | Continuous             |
| TD3       | [`TD3Workflow`](#evorl.algorithms.td3.TD3Workflow)                          | Deterministic | Continuous             |
| SAC       | [`SACWorkflow`](#evorl.algorithms.sac.SACWorkflow)                          | Stochastic    | Discrete & Continuous  |
| TD7       | [`TD7Workflow`](#evorl.algorithms.td7.TD7Workflow)                          | Deterministic | Continuous             |

## Multi-agent RL

MARL workflows use EvoRL's `MultiAgentEnv` API: simultaneous agent-keyed
observations, actions, rewards and done flags. Each agent has its own networks
and optimizer state. Rewards are preserved per agent; environment interaction
counts measure joint environment steps rather than multiplying by agent count.

| Algorithm | Workflow | Critic inputs | Actions |
| --- | --- | --- | --- |
| IPPO | `evorl.algorithms.multi_agents.ippo.IPPOWorkflow` | Local observation | Discrete with masks, continuous Box |
| MAPPO | `evorl.algorithms.multi_agents.mappo.MAPPOWorkflow` | Global state and local observation | Discrete with masks, continuous Box |
| MADDPG | `evorl.algorithms.multi_agents.maddpg.MADDPGWorkflow` | Global state and joint actions | Continuous Box |
| MATD3 | `evorl.algorithms.multi_agents.matd3.MATD3Workflow` | Global state and joint actions, twin critics | Continuous Box |

When an environment provides no `info.global_obs`, centralized critics use
concatenated local observations. Actors always use local observations.
Structured numeric observations are flattened, excluding `action_mask`.
Continuous actions are scaled to each agent's finite Box bounds. Per-agent
termination stops bootstrap; time truncation bootstraps from the original
terminal observation. Inactive-agent transitions are masked out of losses.
MADDPG/MATD3 exploration and target-smoothing noise use normalized `[-1, 1]`
action coordinates before scaling to the environment bounds.

These are feedforward, separate-policy implementations. Recurrent policies,
parameter sharing, PopArt/value normalization and the discrete-action
Gumbel-softmax MADDPG extension are not implemented. The configurations provide
starting points rather than tuned reproductions of the papers' benchmarks.

For example, train MAPPO on discrete MPE or MATD3 on continuous MPE:

```shell
python scripts/train.py agent=exp/multi_agents/mappo-mpe-spread
python scripts/train.py agent=exp/multi_agents/matd3-mpe-spread
```

Algorithm equations and implementation details were checked against
[IPPO](https://arxiv.org/abs/2011.09533),
[MAPPO](https://arxiv.org/abs/2103.01955),
[MADDPG](https://arxiv.org/abs/1706.02275) and
[MATD3](https://arxiv.org/abs/1910.01465).
Implementation references include
[JaxMARL IPPO](https://github.com/FLAIROx/JaxMARL/blob/main/baselines/IPPO/ippo_ff_mpe.py),
[official MAPPO](https://github.com/marlbenchmark/on-policy),
[official MADDPG](https://github.com/openai/maddpg), and
[official MATD3](https://github.com/JohannesAck/MATD3implementation).

## Multi-objective RL

These single-agent workflows learn from vector rewards and nonnegative linear
preferences whose weights sum to one. All objectives are maximized; represent
costs with negative rewards. Observations currently use a flat `Box` space.

| Algorithm | Workflow | Actions |
| --- | --- | --- |
| EnvelopeQ (Envelope Q-learning) | `evorl.algorithms.multi_objective.envelope_q.EnvelopeQWorkflow` | Discrete |
| CAPQL | `evorl.algorithms.multi_objective.capql.CAPQLWorkflow` | Finite continuous Box |

Both use static agents and complete NNX model snapshots in `AgentState`.
Networks receive observations and preference weights; critics return a vector
instead of collapsing rewards into a scalar. Replay preserves reward vectors,
original terminal observations and behavior preferences. EnvelopeQ samples one
behavior preference per episode and relabels replay with additional preferences.
CAPQL samples behavior preferences each step, as in the authors' implementation.
It uses EvoRL's existing TanhNormal policy and a fixed entropy coefficient.

Set `eval_preferences` to a list of normalized weight vectors. With `null`,
evaluation uses each coordinate preference and the balanced preference.
Evaluation reports arrays of preferences, vector returns, scalarized returns
and episode lengths separately. To call a trained policy directly, replace
`agent_state.extra_state.preferences` or supply `SampleBatch.extras.preferences`.
Preference coordinates are never observation-normalized.

```shell
python scripts/train.py agent=exp/multi_objective/envelope_q-deep-sea-treasure
python scripts/train.py agent=exp/multi_objective/capql-brax-ant
python scripts/train.py agent=exp/multi_objective/capql-playground-cartpole
python scripts/train.py agent=exp/multi_objective/envelope_q-mo-deep-sea-treasure
python scripts/train.py agent=exp/multi_objective/capql-mo-mountaincar
```

The EnvelopeQ example includes a native JAX Deep Sea Treasure benchmark with the
classic convex treasure map and standard EvoRL episode/autoreset wrappers.
It lives in `evorl/envs/toy/` for tests and examples, without additional
environment dependencies. The continuous examples reuse
the Brax and MuJoCo Playground vector-reward configurations.
The MO-Gymnasium examples require `pip install -e ".[mo-gymnasium]"`; their
environments run on the CPU while JAX policy evaluation and training can use a GPU.

EnvelopeQ follows the vector squared-L2 and projected absolute-error losses in
Eq. 6/7 of the [original paper](https://proceedings.neurips.cc/paper/2019/hash/4a46fbfca3f1465a27b210f4bdfe6ab3-Abstract.html).
The [released implementation](https://github.com/RunzheYang/MORL) and
[MORL-Baselines](https://github.com/LucasAlegre/morl-baselines) instead square
the projected error. This implementation uses uniform replay and online
selection/target-network evaluation for the envelope Bellman target.
CAPQL follows the vector entropy augmentation and coordinatewise twin-critic
minimum in Algorithm 1 of the
[paper](https://cs.uwaterloo.ca/~y328yu/publication/lu-hy-23/lu-hy-23.pdf),
checked against the [authors' code](https://github.com/haoyelu/CAPQL) and
MORL-Baselines. Configurations are starting points, not tuned benchmark
reproductions.

## EC Algorithms

EC Algorithms are defines in the subpackage `evorl.algorithms.ec`.

Workflows for Single objective EC are derived from [`ECWorkflowTemplate`](#evorl.workflows.ec_workflow.ECWorkflowTemplate).

| Algorithm | Workflow                                                                    | Policy Type   | Supported Action Space |
| --------- | --------------------------------------------------------------------------- | ------------- | ---------------------- |
| OpenES    | [`OpenESWorkflow`](#evorl.algorithms.ec.so.openes.OpenESWorkflow)           | Deterministic | Continuous             |
| VanillaES | [`VanillaESWorkflow`](#evorl.algorithms.ec.so.vanilla_es.VanillaESWorkflow) | Deterministic | Continuous             |
| ARS       | [`ARSWorkflow`](#evorl.algorithms.ec.so.ars.ARSWorkflow)                    | Deterministic | Continuous             |
| CMA-ES    | [`CMAESWorkflow`](#evorl.algorithms.ec.so.cmaes.CMAESWorkflow)              | Deterministic | Continuous             |


Workflows for multi-objective EC derive from [`MultiObjectiveECWorkflowTemplate`](#evorl.workflows.ec_workflow.MultiObjectiveECWorkflowTemplate). [`NSGA2Workflow`](#evorl.algorithms.ec.mo.nsga2.NSGA2Workflow) uses the native [`NSGA2`](#evorl.ec.optimizers.nsga2.NSGA2) optimizer and the standard evaluator. It accepts single-agent environments with vector rewards and the continuous action space supported by the deterministic EC agent. The objective count is inferred from the environment's reward shape; `env.reward_keys` is optional. Without reward names, metrics use `objective_0`, `objective_1`, and so on. For Brax and MuJoCo Playground, select reward components with `env.reward_keys`, for example `env=brax/ant_multi_objective agent=ec/nsga2`. Fitness averages episode returns over episodes and preserves the objective axis. Episode lengths remain a separate statistic.

CMA-ES, SepCMA-ES and NSGA-II implement `EvoOptimizer` directly and maximize fitness without EvoX. NSGA-II maximizes every reward objective; use negative rewards for quantities to minimize. Its variation uses bounded simulated binary crossover and polynomial mutation.

Native NSGA-II requires at least two reward objectives. Its
[`non_dominated_sort`](#evorl.ec.operators.selection.non_dominated.non_dominated_sort)
defaults to `mode="auto"`: CPU uses `sweep` for two objectives and `ens_bs` for
more objectives; GPU and other accelerator backends use `blocked`. The explicit
modes are `sweep`, `ens_ss`, `ens_bs`, and `blocked`. ENS-SS checks fronts
sequentially; ENS-BS locates a front by binary search. The keyword-only
`block_size` parameter controls the blocked path, including automatic GPU
selection, and defaults to 128. It accepts any positive integer and is capped
at the population size. Larger blocks trade additional scratch memory for
potentially fewer batches; the best size depends on the data and hardware.
Sweep and ENS ignore this parameter. All modes support JIT and nested `vmap`.
Mark both `mode` and `block_size` static when supplying them as JIT arguments:

```python
sort = jax.jit(non_dominated_sort, static_argnames=("mode", "block_size"))
ranks = sort(fitnesses, mode="auto", block_size=128)
```

The CMA-ES variants use the positive logarithmic recombination weights from Appendix C of [Hansen's tutorial](https://arxiv.org/abs/1604.00772). Set `num_elites` between 1 and `pop_size // 2`; the supplied configs select half the population. The initial search center is the policy initialized during workflow setup. ES and NSGA-II workflows return the final state from `learn()`, including when the requested iterations are already complete.

### Optional PyTorch EvoX integration

Install the [EvoX extra](../guide/installation.md#optional-evox-algorithms) to use
CSO for scalar rewards or RVEA for vector rewards with at least two objectives:

```shell
python scripts/train.py agent=contrib/cso-evox env=brax/ant
python scripts/train.py agent=contrib/rvea-evox env=brax/ant_multi_objective
```

Both examples use [`EvoXWorkflow`](#evorl.algorithms.contrib.evox.evox_workflow.EvoXWorkflow)
with the deterministic continuous-action EC agent. All reward objectives are
maximized. JAX evaluation and PyTorch optimization share one device through
DLPack. `jit=true` compiles JAX evaluation; `torch_compile=true` independently
compiles PyTorch updates. The outer workflow runs in Python; multi-device
training and outer JAX `jit`/`vmap` are unsupported.

Setup executes EvoX's native `init_step` and counts its sampling. `num_iters`
counts subsequent updates. Only `learn` records metrics and saves checkpoints.
Explicit state includes algorithm tensors, Torch RNG and JAX evaluation data,
so restored state can resume directly.

To add an algorithm, subclass `EvoXWorkflow` and implement `_build_algorithm`
using [`EvoXAlgorithmWrapper`](#evorl.algorithms.contrib.evox.evox_algorithm_wrapper.EvoXAlgorithmWrapper).
Its constructor must support FakeTensor execution; evolving tensors must be
registered parameters or buffers. Supply fixed, unregistered tensor attributes
through `static_tensors`. 64-bit tensors are rejected; the RVEA example uses
int32 generation counters.

## ERL Algorithms

The ERL algorithms are defined in the subpackage [`evorl.algorithms.erl`](#evorl.algorithms.erl). We provide ERL and CEM-RL and their variants.

## Population-based AutoRL Algorithms

The Population-based AutoRL algorithms are defined in the subpackage [`evorl.algorithms.meta`](#evorl.algorithms.meta). We provide some general population-based training pipelines for RL hyperparameter tuning.
