# Algorithms

Currently, EvoRL supports various training pipelines (workflows):

1. Reinforcement Learning (RL) Algorithms
2. Evolutionary Computation (EC) Algorithms, specific for policy search
3. Evolutionary Reinforcement Learning (EvoRL):
    - Evolution-guided Reinforcement Learning (ERL)
    - Population-based AutoRL

This document introduces these types of algorithms implemented in EvoRL. All algorithms are defined in [`evorl.algorithms`](#evorl.algorithms).

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

The CMA-ES variants use the positive logarithmic recombination weights from Appendix C of [Hansen's tutorial](https://arxiv.org/abs/1604.00772). Set `num_elites` between 1 and `pop_size // 2`; the supplied configs select half the population. The initial search center is the policy initialized during workflow setup. ES and NSGA-II workflows return the final state from `learn()`, including when the requested iterations are already complete.

## ERL Algorithms

The ERL algorithms are defined in the subpackage [`evorl.algorithms.erl`](#evorl.algorithms.erl). We provide ERL and CEM-RL and their variants.

## Population-based AutoRL Algorithms

The Population-based AutoRL algorithms are defined in the subpackage [`evorl.algorithms.meta`](#evorl.algorithms.meta). We provide some general population-based training pipelines for RL hyperparameter tuning.
