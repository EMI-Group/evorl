"""MORL/D with fixed-preference PPO subproblems and neighborhood cooperation.

This is the linear weighted-sum, SER instantiation of Felten et al. (2024),
https://doi.org/10.1613/jair.1.15702. Subproblems take turns interacting with
the environment; each keeps independent model and optimizer state. Unlike the
MORL-Baselines MOSAC instantiation, this on-policy variant has no shared replay.
"""

import copy
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from evorl.metrics import TrainMetric
from evorl.recorders import add_prefix
from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import scan_and_last
from evorl.workflows import Workflow

from .common import MultiObjectiveEvaluateMetric, MultiObjectiveWorkflowMetric
from .moppo import MOPPOWorkflow
from .morld_utils import (
    decomposition_weights,
    nearest_decomposition_neighbors,
    psa_preferences,
    reset_decomposition_env,
    update_decomposition_archive,
)


def _evaluate_policies(workflow, population, preferences):
    metrics, population = jax.vmap(workflow.evaluate)(population)
    return MultiObjectiveEvaluateMetric(
        preferences=preferences,
        episode_return=metrics.episode_return,
        scalarized_return=(preferences * metrics.episode_return).sum(-1),
        episode_length=metrics.episode_length,
    ), population


class MORLDPPOWorkflow(Workflow):
    """Solve weighted-sum PPO subproblems with round-robin policy improvement.

    First-round transfer copies an improved actor to untrained neighbors and
    resets their optimizers. PSA can subsequently adapt decomposition weights.
    A complete training chunk starts a new episode batch, matching the
    reference's per-policy training calls. All population, archive and RNG data
    reside in the returned `State`, including for checkpoint restoration.

    Devices split the selected policy's environment and minibatch axes, while
    its parameters and optimizer are replicated. Gradient averaging is only
    across replicas of the same policy. It optimizes SER with linear utility;
    it does not implement ESR or replay sharing. The compiled step can itself
    be nested under `jax.vmap`.
    """

    def __init__(self, workflow, config):
        super().__init__(config)
        self.workflow = workflow
        self.mesh = workflow.mesh
        self.steps_per_exchange = config.exchange_every // (
            workflow.config.num_envs
            * workflow.config.rollout_length
            * workflow.num_devices
        )
        weights = config.preferences
        if weights is None:
            weights = decomposition_weights(
                workflow.agent.num_objectives, config.pop_size
            )
        weights = np.asarray(weights, np.float32)
        if (
            weights.shape != (config.pop_size, workflow.agent.num_objectives)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.allclose(weights.sum(-1), 1, atol=1e-6)
        ):
            raise ValueError(
                "preferences must contain one normalized vector per policy"
            )
        self.initial_preferences = tuple(map(tuple, weights.tolist()))

    @classmethod
    def name(cls):
        return "MORL/D-PPO"

    @classmethod
    def build_from_config(cls, config, multi_device=False, jit=True):
        config = copy.deepcopy(config)
        if config.pop_size < 2:
            raise ValueError("pop_size must be at least two")
        if not 0 <= config.neighborhood_size < config.pop_size:
            raise ValueError("neighborhood_size must be between zero and pop_size - 1")
        if config.archive_capacity < config.pop_size:
            raise ValueError("archive_capacity must fit the initial population")
        if config.total_timesteps < 0 or config.eval_interval <= 0:
            raise ValueError(
                "Sampling budget must be nonnegative and eval_interval positive"
            )
        if config.weight_adaptation not in (None, "PSA") or config.psa_delta < 0:
            raise ValueError(
                "weight_adaptation must be null or PSA and psa_delta nonnegative"
            )
        inner_config = OmegaConf.merge(
            config.target_workflow,
            dict(
                env=OmegaConf.to_container(config.env, resolve=True),
                checkpoint=dict(enable=False),
                output_dir=config.output_dir,
            ),
        )
        stride = inner_config.num_envs * inner_config.rollout_length
        if stride <= 0 or config.exchange_every <= 0 or config.exchange_every % stride:
            raise ValueError(
                "exchange_every must be a positive multiple of one inner rollout"
            )
        policy_cls = type(f"{cls.__name__}PolicyWorkflow", (MOPPOWorkflow,), {})
        workflow = policy_cls.build_from_config(
            inner_config, multi_device=multi_device, jit=jit
        )
        OmegaConf.set_readonly(config, True)
        result = cls(workflow, config)
        if jit:
            cls.setup = jax.jit(cls.setup, static_argnums=(0,))
            cls.step = jax.jit(cls.step, static_argnums=(0, 2))
            cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        return result

    def setup(self, key):
        key, population_key = jax.random.split(key)
        population = jax.vmap(self.workflow.setup)(
            jax.random.split(population_key, self.config.pop_size)
        )
        preferences = jnp.asarray(self.initial_preferences, jnp.float32)
        population = population.replace(
            agent_state=population.agent_state.replace(
                extra_state=population.agent_state.extra_state.replace(
                    preferences=preferences
                )
            )
        )
        evaluation, population = _evaluate_policies(
            self.workflow, population, preferences
        )
        capacity = self.config.archive_capacity
        archive = PyTreeDict(
            returns=jnp.zeros((capacity, self.workflow.agent.num_objectives)),
            valid=jnp.zeros(capacity, jnp.bool_),
            preferences=jnp.zeros((capacity, self.workflow.agent.num_objectives)),
            agent_states=jax.tree.map(
                lambda x: jnp.broadcast_to(x[0], (capacity, *x.shape[1:])),
                population.agent_state,
            ),
        )
        archive = update_decomposition_archive(
            archive, evaluation.episode_return, population.agent_state, preferences
        )
        return State(
            key=key,
            population=population,
            preferences=preferences,
            archive=archive,
            evaluation=evaluation,
            current_policy=jnp.int32(0),
            metrics=MultiObjectiveWorkflowMetric(),
        )

    def step(self, state, num_steps=None):
        num_steps = self.steps_per_exchange if num_steps is None else num_steps
        if not 0 < num_steps <= self.steps_per_exchange:
            raise ValueError("num_steps must be in [1, steps_per_exchange]")
        index = state.current_policy
        key, reset_key = jax.random.split(state.key)
        member = jax.tree.map(lambda x: x[index], state.population)
        previous_metrics = member.metrics
        member = member.replace(
            env_state=reset_decomposition_env(self.workflow, reset_key)
        )

        def improve(member, _):
            train, member = self.workflow.step(member)
            return member, train

        member, train = scan_and_last(improve, member, (), length=num_steps)
        population = jax.tree.map(
            lambda members, updated: members.at[index].set(updated),
            state.population,
            member,
        )
        evaluation, population = _evaluate_policies(
            self.workflow, population, state.preferences
        )
        archive = update_decomposition_archive(
            state.archive,
            evaluation.episode_return,
            population.agent_state,
            state.preferences,
        )
        neighbors = nearest_decomposition_neighbors(
            state.preferences, self.config.neighborhood_size
        )[index]
        if self.config.transfer:

            def transfer(carry, neighbor):
                agents, optimizers = carry
                target = jax.tree.map(lambda x: x[neighbor], agents)
                optimizer = jax.tree.map(lambda x: x[neighbor], optimizers)

                def copy_actor(carry):
                    agent, optimizer = carry
                    models = self.workflow.agent.get_models(agent, copy=True)
                    source = self.workflow.agent.get_models(
                        member.agent_state, copy=True
                    )
                    models.policy = nnx.clone(source.policy)
                    optimizer = nnx.unpack(
                        nnx.Optimizer(models, self.workflow.optimizer, wrt=nnx.Param)
                    )
                    agent = agent.replace(
                        model_state=nnx.unpack(models),
                        obs_preprocessor_state=member.agent_state.obs_preprocessor_state,
                    )
                    return agent, optimizer

                target, optimizer = jax.lax.cond(
                    (state.metrics.iterations < self.config.pop_size)
                    & (neighbor > index),
                    copy_actor,
                    lambda carry: carry,
                    (target, optimizer),
                )
                agents = jax.tree.map(
                    lambda x, y: x.at[neighbor].set(y), agents, target
                )
                optimizers = jax.tree.map(
                    lambda x, y: x.at[neighbor].set(y), optimizers, optimizer
                )
                return (agents, optimizers), None

            (agents, optimizers), _ = jax.lax.scan(
                transfer, (population.agent_state, population.opt_state), neighbors
            )
            population = population.replace(agent_state=agents, opt_state=optimizers)
        preferences = state.preferences
        if self.config.weight_adaptation == "PSA":
            preferences = psa_preferences(
                preferences,
                evaluation.episode_return,
                archive.returns,
                archive.valid,
                self.config.psa_delta,
            )
            population = population.replace(
                agent_state=population.agent_state.replace(
                    extra_state=population.agent_state.extra_state.replace(
                        preferences=preferences
                    )
                )
            )
        metrics = state.metrics.replace(
            iterations=state.metrics.iterations + 1,
            sampled_timesteps=state.metrics.sampled_timesteps
            + member.metrics.sampled_timesteps
            - previous_metrics.sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes
            + member.metrics.sampled_episodes
            - previous_metrics.sampled_episodes,
            training_updates=state.metrics.training_updates
            + member.metrics.training_updates
            - previous_metrics.training_updates,
        )
        return TrainMetric(
            loss=train.loss,
            raw_loss_dict=PyTreeDict(
                inner=train.raw_loss_dict,
                policy_index=index,
                archive_size=archive.valid.sum(),
            ),
        ), state.replace(
            key=key,
            population=population,
            preferences=preferences,
            archive=archive,
            evaluation=evaluation,
            current_policy=(index + 1) % self.config.pop_size,
            metrics=metrics,
        )

    def evaluate(self, state):
        evaluation, population = _evaluate_policies(
            self.workflow, state.population, state.preferences
        )
        return evaluation, state.replace(population=population, evaluation=evaluation)

    def learn(self, state):
        stride = (
            self.workflow.config.num_envs
            * self.workflow.config.rollout_length
            * self.num_devices
        )
        while int(state.metrics.sampled_timesteps) < self.config.total_timesteps:
            remaining = self.config.total_timesteps - int(
                state.metrics.sampled_timesteps
            )
            steps = min(self.steps_per_exchange, math.ceil(remaining / stride))
            train, state = self.step(state, steps)
            iteration = int(state.metrics.iterations)
            final = int(state.metrics.sampled_timesteps) >= self.config.total_timesteps
            self.recorder.write(state.metrics.to_local_dict(), iteration)
            self.recorder.write(train.to_local_dict(), iteration)
            if iteration % self.config.eval_interval == 0 or final:
                evaluation, state = self.evaluate(state)
                self.recorder.write(
                    add_prefix(evaluation.to_local_dict(), "eval"), iteration
                )
            self.checkpoint_manager.save(iteration, state, force=final)
        return state

    def close(self):
        try:
            self.workflow.close()
        finally:
            super().close()
