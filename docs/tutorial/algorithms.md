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

## EC Algorithms

EC Algorithms are defines in the subpackage `evorl.algorithms.ec`.

Workflows for Single objective EC are derived from [`ECWorkflowTemplate`](#evorl.workflows.ec_workflow.ECWorkflowTemplate).

| Algorithm | Workflow                                                                    | Policy Type   | Supported Action Space |
| --------- | --------------------------------------------------------------------------- | ------------- | ---------------------- |
| OpenES    | [`OpenESWorkflow`](#evorl.algorithms.ec.so.openes.OpenESWorkflow)           | Deterministic | Continuous             |
| VanillaES | [`VanillaESWorkflow`](#evorl.algorithms.ec.so.vanilla_es.VanillaESWorkflow) | Deterministic | Continuous             |
| ARS       | [`ARSWorkflow`](#evorl.algorithms.ec.so.ars.ARSWorkflow)                    | Deterministic | Continuous             |
| CMA-ES    | [`CMAESWorkflow`](#evorl.algorithms.ec.so.cmaes.CMAESWorkflow)              | Deterministic | Continuous             |


Workflows for Multi-objective EC are derived from [`MultiObjectiveECWorkflowTemplate`](#evorl.workflows.ec_workflow.MultiObjectiveECWorkflowTemplate). Currently, we provide NSGA-II with [`NSGA2Workflow`](#evorl.algorithms.ec.mo.nsga2_brax.NSGA2Workflow) for brax environments.

## ERL Algorithms

The ERL algorithms are defined in the subpackage [`evorl.algorithms.erl`](#evorl.algorithms.erl). We provide ERL and CEM-RL and their variants.

## Population-based AutoRL Algorithms

The Population-based AutoRL algorithms are defined in the subpackage [`evorl.algorithms.meta`](#evorl.algorithms.meta). We provide some general population-based training pipelines for RL hyperparameter tuning.
