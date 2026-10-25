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

PPO, ParamPPO and MOPPO normalize advantages over each global training minibatch,
excluding autoreset transitions. Losses are weighted by valid sample counts,
including when those counts differ between devices. PBT members remain independent.
All three use ENVPOOL (next-step) autoreset and the common GAE implementation.
Sampling budgets count rollout steps, including reset-only slots.
PPO-family and IMPALA KL diagnostics use the common K3 estimator of
`KL(behavior || current)`, excluding invalid transitions.

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
IPPO/MAPPO use ENVPOOL next-step autoreset: reset-only transitions are masked
for every agent, and adjacent local/global observations retain the terminal
values needed for truncation bootstrap.
MADDPG/MATD3 exploration and target-smoothing noise use normalized `[-1, 1]`
action coordinates before scaling to the environment bounds.

These are feedforward, separate-policy implementations. Recurrent policies,
parameter sharing, PopArt/value normalization and the discrete-action
Gumbel-softmax MADDPG extension are not implemented. The configurations provide
starting points rather than tuned reproductions of the papers' benchmarks.

Each agent's reward is scalar. All four `step` methods support JIT and data
parallelism: environments are sharded, agent models/optimizers are replicated,
and gradients synchronize only replicas of the same agent. MADDPG/MATD3 keep a
device-local ring of aligned joint transitions.

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

These single-agent workflows maximize vector rewards; represent costs with
negative rewards. Observations use a flat `Box` space. Preference weights are
nonnegative and sum to one. SER applies utility to the expected return; ESR
averages episode utilities. They coincide for linear utility.

CAPQL, MOPPO and MOSAC use normalized continuous actions in `[-1, 1]`.
Use `ActionSquashWrapper` around environments with other native Box bounds;
the agents do not rescale actions or apply an additional density correction.

| Algorithm | Workflow | Policy family | Criterion | Actions |
| --- | --- | --- | --- | --- |
| EnvelopeQ | `EnvelopeQWorkflow` | Preference-conditioned | SER | Discrete |
| CAPQL | `CAPQLWorkflow` | Preference-conditioned | SER | Finite continuous Box |
| GPI-LS | `GPILSWorkflow` | Successor features and GPI | SER | Discrete |
| PCN | `PCNWorkflow` | Return/horizon-conditioned | Deterministic dynamics/rewards | Discrete or finite continuous Box |
| MOPPO | `MOPPOWorkflow` | One fixed-preference policy | SER | Discrete or finite continuous Box |
| MOSAC | `MOSACWorkflow` | One fixed-preference policy | SER | Discrete or finite continuous Box |
| PGMORL | `PGMORLWorkflow` | Independent PPO policies | SER | Finite continuous Box |
| MORL/D-PPO | `MORLDPPOWorkflow` | Independent PPO subproblems | SER, linear utility | Discrete or finite continuous Box |
| MORL/D-SAC | `MORLDSACWorkflow` | Independent SAC subproblems | SER, linear utility | Discrete or finite continuous Box |

Workflows live in `evorl.algorithms.multi_objective`. EnvelopeQ, CAPQL, GPI-LS
and PCN each share one model across multiple represented policies. PGMORL and
MORL/D maintain separate models and optimizers per population member.
Configurations are starting points, not tuned benchmark reproductions.

### Algorithm details and scope

- **EnvelopeQ** uses the vector squared-L2 and projected absolute-error losses
  from Eq. 6/7 of the [paper](https://proceedings.neurips.cc/paper/2019/hash/4a46fbfca3f1465a27b210f4bdfe6ab3-Abstract.html).
  The [released code](https://github.com/RunzheYang/MORL) and MORL-Baselines
  instead square the projected error. Behavior preferences persist for an
  episode; replay is relabeled with additional preferences. Bellman targets use
  online selection and target-network evaluation.
- **CAPQL** follows the [paper](https://cs.uwaterloo.ca/~y328yu/publication/lu-hy-23/lu-hy-23.pdf)
  and [authors' code](https://github.com/haoyelu/CAPQL): vector entropy augmentation,
  coordinatewise twin-critic minimum, per-step behavior preferences, a TanhNormal
  policy and fixed entropy coefficient.
- **GPI-LS** uses policy-conditioned DDQN with exact linear-support corners.
  Corner enumeration grows combinatorially; `max_support_size` bounds storage
  and raises when exhausted. Ensemble/prioritized replay and GPI-PD are excluded.
- **PCN** learns actions by supervised hindsight on complete episodes, conditioned
  on desired return and remaining horizon. Padded transitions are excluded from
  losses and counters; sampling budgets finish after a whole episode batch.
  `scaling_factor` scales the commands. For discounted returns, the next command
  is `(command - reward) / discount`. Optional `normalize_obs` defaults to
  `false`; observation statistics accumulate only real collected transitions.

  PCN assumes that state and action determine the next state and reward;
  random resets and exploratory actions are allowed. Native DeepSeaTreasure
  satisfies this assumption; Brax/Playground tasks may satisfy it with fixed
  physics and perturbations disabled. Slippery FrozenLake and stochastic rewards
  do not: hindsight imitation can learn lucky outcomes rather than expected
  returns. Reusing an RNG seed does not make a stochastic task deterministic.
- **MOPPO** uses vector values and GAE, then scalarizes the actor advantage.
  PGMORL and MORL/D-PPO use this same PPO training workflow.
  **MOSAC** scalarizes each vector critic before the twin-critic minimum and
  regresses a scalar Bellman target. MOSAC supports optional temperature tuning
  and exact categorical expectations for discrete actions. It uses SAC's
  action distribution, alpha loss, optimizer configuration and replay fields.
  `actor_update_interval` counts critic updates per actor update for either
  action type; the target updates once afterward. `num_updates_per_iter` counts
  these update cycles. Agents use normalized actions; `ActionSquashWrapper`
  handles native Box bounds. Its `update(state)`
  trains from replay without collecting or counting new environment samples.
- **PGMORL** selects PPO tasks using a directional performance buffer and local
  predictions. It supports two or three objectives and excludes the final
  Pareto-analysis/interpolation stage, matching MORL-Baselines.
  Host environments finish whole population batches, so the last batch can
  exceed `total_timesteps`; all collected transitions are counted.
- **MORL/D-PPO and MORL/D-SAC** use round-robin collection, first-round actor
  transfer and optional PSA preference adjustment. Pareto archives preserve
  policy snapshots evaluated with the training discount, as does PGMORL's
  performance model. These variants implement weighted-sum SER; Tchebycheff and
  nonlinear ESR training are excluded.

The additional implementations follow [MORL-Baselines](https://github.com/LucasAlegre/morl-baselines).
References: [GPI-LS](https://arxiv.org/abs/2301.07784),
[PCN](https://arxiv.org/abs/2204.05036),
[PGMORL](https://proceedings.mlr.press/v119/xu20h.html),
[MORL/D](https://doi.org/10.1613/jair.1.15702).

### Replay and execution

| Algorithms | Training data | Retention and sampling |
| --- | --- | --- |
| EnvelopeQ, CAPQL, GPI-LS, MOSAC | Transition ring | Overwrite oldest entries; uniform sampling |
| PCN | Complete-episode archive | Retain Pareto quality/diversity; sample an episode uniformly, then one valid timestep |
| MOPPO, PGMORL, MORL/D-PPO | Current PPO rollout | Reuse for PPO epochs, then discard; no replay across iterations |
| MORL/D-SAC | MOSAC transition rings | Default `shared_replay_buffer=true`: one shared ring; `false`: one ring per member |

Transition replay preserves vector rewards, original terminal observations and
termination flags in `extras.env_extras.ori_obs` and `.termination`, as in SAC.
EnvelopeQ, CAPQL and GPI-LS also store behavior preferences;
MOSAC applies its learner's current preference during loss computation. These
rings use uniform replay, not prioritized sampling. PCN's retention priorities
control which episodes survive, not their training-sampling probabilities.
PGMORL's performance buffer and Pareto archive store policy snapshots and
performance data, rather than transition replay.

| Workflow | JIT boundary | Multiple devices |
| --- | --- | --- |
| EnvelopeQ, CAPQL, MOPPO, MOSAC | Whole `step` | Split environments and batches; replicated models/optimizers and averaged gradients; off-policy replay stays local |
| GPI-LS | Training/evaluation; support selection stays in Python | Same data parallelism as above |
| PCN | Whole `step` | Replicated global episode archive; gather complete episodes before common retention and sample local batches from the full archive |
| PGMORL | Population PPO training/evaluation; prediction/selection stay in Python | Single device |
| MORL/D-PPO | Whole `step` | Data parallel within each member; keep round-robin collection |
| MORL/D-SAC | Whole `step` | Data parallel within each member; replay stays on the collecting device |

MORL/D-SAC shares each device's local ring among policy replicas when
`shared_replay_buffer=true`; otherwise each policy has its own local ring.
Only collection writes replay. `update_passes` controls extra replay-only updates
to the other members. Shared warmup is collected and counted once; private
warmup is counted per member. Actor transfer preserves replay, critic/target
parameters, temperature and counters, and transfers observation statistics.
Replay prefill retains its transitions and observation statistics; training
starts from reset environments rather than continuing the prefill episode.

Environment counts and training batch sizes are configured as single-device
totals and divided across devices, as is transition replay capacity per learner.
PCN keeps episode capacity, initial episode count and command candidate counts
global. MORL/D keeps `exchange_every` global and its Pareto policy archive
replicated. Gradients are never averaged between different population members.
External nested `vmap` creates independent workflow states, including independent
replay/archive ownership; PGMORL's Python outer loop is excluded.

Playground population examples select JAX physics because Warp FFI does not
support their nested batching. With MJX 3.16.0 and Playground 0.2.0, VMA checks
under `shard_map` fail in the Cartpole/Cheetah line search and Pendulum physics
step. Ordinary PPO also reproduces these failures; they are not specific to
MORL/D or multi-objective losses. Workflow VMA checks remain enabled.

### Configuration and examples

For EnvelopeQ, CAPQL and GPI-LS, `eval_preferences=null` evaluates coordinate
preferences and the balanced preference. Evaluation reports preference/command
vectors, vector returns and episode lengths separately. To select a preference
for an action, set `agent_state.extra_state.preferences` or
`SampleBatch.extras.preferences`; preference coordinates are not observation-normalized.

Experiment YAMLs are grouped by algorithm under `configs/agent/exp/multi_objective/`:

```shell
python scripts/train.py agent=exp/multi_objective/envelope_q/deep-sea-treasure
python scripts/train.py agent=exp/multi_objective/pcn/deep-sea-treasure
python scripts/train.py agent=exp/multi_objective/morld_sac/brax-ant
```

Native DeepSeaTreasure lives in `evorl/envs/toy/` and needs no optional environment
library. Brax/Playground examples select reward components through their
environment YAMLs. MO-Gymnasium examples require
`uv pip install -e ".[mo-gymnasium]"`; simulation runs on the CPU while JAX models
can train and evaluate on a GPU.

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

The core [`evorl.algorithms.erl`](#evorl.algorithms.erl) workflows are
`CEMRLWorkflow`, `CEMRLOpenESWorkflow`, `ERLGAWorkflow` and `ERLESWorkflow`.
CEMRL and CEMRL-OpenES share `CEMRLWorkflowBase` for shared-critic training,
initialization and evaluation, while defining their own evolutionary steps.
ERL-GA/ES maintain independent TD3 learners alongside the evolutionary population.
Both families use shared replay.

Original-paper variants and ERL-EDA live in
`evorl.algorithms.contrib`, with configurations under `agent=contrib/...`.
`PopEpisodicTD3Workflow` is an independent TD3 population with no evolutionary
optimizer or state. These workflows share episode collection and replay
initialization functions; shared-critic and independent-critic updates stay separate.
RL updates use direct NNX update functions, with model and optimizer snapshots
restored once per update. Subclasses select the evolutionary optimizer through
`_build_ec_optimizer`; each `step()` defines its collection, learning and injection
order. Original-paper variants choose their own runtime update budgets.

## Population-based AutoRL Algorithms

The Population-based AutoRL algorithms are defined in the subpackage [`evorl.algorithms.meta`](#evorl.algorithms.meta). We provide some general population-based training pipelines for RL hyperparameter tuning.

PBT-SAC shards independent SAC members along `META_POP`, rather than applying
data-parallel gradient averaging to one SAC policy. Each device vmaps the members
assigned to it. With the default `shared_replay_buffer=true`, a complete replay
ring is replicated on every device, and new member trajectories are explicitly
`all_gather`ed before identical writes to those replicas. All members
can therefore sample the whole population's experience. Setting
`shared_replay_buffer=false` gives every member an independent ring, sharded
with the population; members write and sample only their own experience, with
no replay exchange between devices. Capacity and batch size apply to each ring.
Warmup collects `random_timesteps` transitions per ring, rounded down to a full
environment batch. Replay is owned by the outer PBT state and is not copied
during exploit in either mode. This differs from ordinary data-parallel
SAC/MOSAC, which keeps a local ring per device and synchronizes gradients for
the same policy.
Exploitation retains each recipient's root random key, keeping parent and
offspring exploration independent after model/environment copying.
