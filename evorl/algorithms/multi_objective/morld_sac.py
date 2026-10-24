"""MORL/D-SAC with fixed-preference MOSAC subproblems.

Policies train in round-robin order, with optional extra replay-only updates
for the other policies. A shared ring is owned once by the outer state; private
rings are owned by their respective population members. Replay always stores
the original reward vector and terminal observation, never a scalarized reward.
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
from evorl.utils.jax_utils import scan_and_last, rng_fold_in_axis
from evorl.workflows import Workflow

from .common import MultiObjectiveEvaluateMetric, MultiObjectiveWorkflowMetric
from .morld_utils import (
    decomposition_weights,
    nearest_decomposition_neighbors,
    psa_preferences,
    reset_decomposition_env,
    update_decomposition_archive,
)
from .mosac import MOSACWorkflow


class MORLDSACWorkflow(Workflow):
    """Coordinate independent MOSAC policies with shared or private replay.

    `shared_replay_buffer=True` stores one replay ring in `State.replay_buffer_state`;
    member replay fields are `None`. All policies can sample from this ring,
    using independent keys and their own preferences. Only the selected policy
    collects transitions. With `shared_replay_buffer=False`, each member instead owns
    its own ring and samples only that ring. `update_passes` controls replay-only
    updates to the other members after each training chunk.
    Each pass calls MOSAC's `update`, which runs
    `target_workflow.num_updates_per_iter` critic updates per member.

    Neighbor transfer copies only actor parameters and their observation
    statistics, resetting the recipient actor optimizer. Critics, targets,
    temperature, RNG keys, counters and replay remain with their owners.
    With observation normalization enabled, both actor and critic subsequently
    use the transferred observation statistics; critic parameters are retained.

    Devices split each policy's environments and replay capacity. Parameters
    are replicated, with gradients averaged only across the same policy's
    replicas. Shared replay is shared among local policy replicas on each
    device; private replay gives each policy its own local ring. Transitions
    stay on the collecting device. Setup, step and evaluation support JIT and
    nested `jax.vmap`. Objectives use weighted-sum SER.
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
        return "MORL/D-SAC"

    @classmethod
    def build_from_config(cls, config, multi_device=False, jit=True):
        config = copy.deepcopy(config)
        if config.pop_size < 2:
            raise ValueError("pop_size must be at least two")
        if not 0 <= config.neighborhood_size < config.pop_size:
            raise ValueError("neighborhood_size must be between zero and pop_size - 1")
        if config.archive_capacity < config.pop_size:
            raise ValueError("archive_capacity must fit the initial population")
        if (
            config.total_timesteps < 0
            or config.eval_interval <= 0
            or config.update_passes < 0
        ):
            raise ValueError(
                "Budgets and update_passes must be nonnegative and eval_interval positive"
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
        policy_cls = type(f"{cls.__name__}PolicyWorkflow", (MOSACWorkflow,), {})
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
        key, population_key, replay_key = jax.random.split(key, 3)
        preferences = jnp.asarray(self.initial_preferences, jnp.float32)
        population_keys = jax.random.split(population_key, self.config.pop_size)
        if self.config.shared_replay_buffer:
            # Construct learners without allocating or warming up a replay ring
            # for each one. The outer workflow owns the only shared ring.
            def initialize(member_key, preference):
                member_key, agent_key, env_key = jax.random.split(member_key, 3)
                agent, optimizer = self.workflow._setup_agent_and_optimizer(agent_key)
                agent = agent.replace(
                    extra_state=agent.extra_state.replace(preferences=preference)
                )
                metrics = MultiObjectiveWorkflowMetric()
                if self.workflow.is_multi_device:
                    member_key, agent, optimizer, metrics = jax.device_put(
                        (member_key, agent, optimizer, metrics), jax.P()
                    )
                return State(
                    key=member_key,
                    agent_state=agent,
                    opt_state=optimizer,
                    env_state=reset_decomposition_env(self.workflow, env_key),
                    replay_buffer_state=None,
                    metrics=metrics,
                )

            population = jax.vmap(initialize)(population_keys, preferences)
            member = jax.tree.map(lambda x: x[0], population)
            replay_key, warmup_env_key = jax.random.split(replay_key)
            # A host adapter's population reset creates P*N native environments.
            # Warmup collects for one member, so reset that batch explicitly;
            # every later round-robin training chunk does the same.
            member = member.replace(
                env_state=reset_decomposition_env(self.workflow, warmup_env_key)
            )
            if self.workflow.is_multi_device:

                def init_replay(key):
                    key = rng_fold_in_axis(key, self.workflow.dp_axis_name)
                    replay = self.workflow._setup_replaybuffer(key)
                    return replay.replace(
                        current_index=replay.current_index[None],
                        buffer_size=replay.buffer_size[None],
                    )

                replay = jax.shard_map(
                    init_replay,
                    in_specs=jax.P(),
                    out_specs=jax.P(self.workflow.dp_axis_name),
                )(replay_key)
            else:
                replay = self.workflow._setup_replaybuffer(replay_key)
            member = self.workflow._postsetup_replaybuffer(
                member.replace(replay_buffer_state=replay)
            )
            replay = member.replay_buffer_state
            member = member.replace(replay_buffer_state=None)
            population = jax.tree.map(lambda x, y: x.at[0].set(y), population, member)
            if member.agent_state.obs_preprocessor_state is not None:
                statistics = jax.tree.map(
                    lambda x: jnp.broadcast_to(x, (self.config.pop_size, *x.shape)),
                    member.agent_state.obs_preprocessor_state,
                )
                population = population.replace(
                    agent_state=population.agent_state.replace(
                        obs_preprocessor_state=statistics
                    )
                )
        else:
            population = jax.vmap(self.workflow.setup)(population_keys)
            replay = None
        population = population.replace(
            agent_state=population.agent_state.replace(
                extra_state=population.agent_state.extra_state.replace(
                    preferences=preferences
                )
            )
        )
        evaluation, population = jax.vmap(self.workflow.evaluate)(population)
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
            archive, evaluation.episode_returns, population.agent_state, preferences
        )
        return State(
            key=key,
            population=population,
            replay_buffer_state=replay,
            preferences=preferences,
            archive=archive,
            current_policy=jnp.int32(0),
            metrics=MultiObjectiveWorkflowMetric(
                sampled_timesteps=population.metrics.sampled_timesteps.sum(),
                sampled_episodes=population.metrics.sampled_episodes.sum(),
            ),
        )

    def step(self, state, num_steps=None):
        num_steps = self.steps_per_exchange if num_steps is None else num_steps
        if not 0 < num_steps <= self.steps_per_exchange:
            raise ValueError("num_steps must be in [1, steps_per_exchange]")
        index = state.current_policy
        key, reset_key = jax.random.split(state.key)
        member = jax.tree.map(lambda x: x[index], state.population)
        if self.workflow.is_multi_device and not self.config.shared_replay_buffer:
            # A gather can drop an Explicit axis on singleton cursor dimensions
            # when the mesh has one device. Restore the member boundary layout.
            member = member.replace(
                replay_buffer_state=jax.reshard(
                    member.replay_buffer_state, jax.P(self.workflow.dp_axis_name)
                )
            )
        before = member.metrics
        member = member.replace(
            env_state=reset_decomposition_env(self.workflow, reset_key)
        )
        if self.config.shared_replay_buffer:
            member = member.replace(replay_buffer_state=state.replay_buffer_state)

        def improve(member, _):
            metrics, member = self.workflow.step(member)
            return member, metrics

        member, train = scan_and_last(improve, member, (), length=num_steps)
        replay = (
            member.replay_buffer_state if self.config.shared_replay_buffer else None
        )
        if self.config.shared_replay_buffer:
            member = member.replace(replay_buffer_state=None)
        population = jax.tree.map(
            lambda x, y: x.at[index].set(y), state.population, member
        )
        other_update_loss = jnp.float32(0)
        if self.config.update_passes:
            other_indices = (
                index + 1 + jnp.arange(self.config.pop_size - 1)
            ) % self.config.pop_size
            others = jax.tree.map(lambda x: x[other_indices], population)

            def update(member):
                if self.config.shared_replay_buffer:
                    member = member.replace(replay_buffer_state=replay)
                elif self.workflow.is_multi_device:
                    member = member.replace(
                        replay_buffer_state=jax.reshard(
                            member.replay_buffer_state,
                            jax.P(self.workflow.dp_axis_name),
                        )
                    )
                metrics, member = self.workflow.update(member)
                if self.config.shared_replay_buffer:
                    member = member.replace(replay_buffer_state=None)
                return member, metrics

            def update_others(others, _):
                return jax.vmap(update)(others)

            others, other_metrics = scan_and_last(
                update_others, others, (), length=self.config.update_passes
            )
            population = jax.tree.map(
                lambda x, y: x.at[other_indices].set(y), population, others
            )
            other_update_loss = other_metrics.loss.mean()

        evaluation, population = jax.vmap(self.workflow.evaluate)(population)
        archive = update_decomposition_archive(
            state.archive,
            evaluation.episode_returns,
            population.agent_state,
            state.preferences,
        )
        if self.config.transfer:
            neighbors = nearest_decomposition_neighbors(
                state.preferences, self.config.neighborhood_size
            )[index]

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
                    models.actor = nnx.clone(source.actor)
                    optimizer = optimizer.replace(
                        actor=nnx.unpack(
                            nnx.Optimizer(
                                models.actor,
                                self.workflow.optimizer,
                                wrt=nnx.Param,
                            )
                        )
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
                evaluation.episode_returns,
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
            - before.sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes
            + member.metrics.sampled_episodes
            - before.sampled_episodes,
            training_updates=population.metrics.training_updates.sum(),
        )
        return TrainMetric(
            loss=train.loss,
            raw_loss_dict=PyTreeDict(
                selected=train.raw_loss_dict,
                other_update_loss=other_update_loss,
                policy_index=index,
                archive_size=archive.valid.sum(),
            ),
        ), state.replace(
            key=key,
            population=population,
            replay_buffer_state=replay,
            preferences=preferences,
            archive=archive,
            current_policy=(index + 1) % self.config.pop_size,
            metrics=metrics,
        )

    def evaluate(self, state):
        evaluation, population = jax.vmap(self.workflow.evaluate)(state.population)
        preferences = state.preferences
        result = MultiObjectiveEvaluateMetric(
            preferences=preferences,
            episode_returns=evaluation.episode_returns,
            scalarized_returns=(preferences * evaluation.episode_returns).sum(-1),
            episode_lengths=evaluation.episode_lengths,
        )
        return result, state.replace(population=population)

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
            train, state = self.step(
                state, min(self.steps_per_exchange, math.ceil(remaining / stride))
            )
            iteration = int(state.metrics.iterations)
            complete = (
                int(state.metrics.sampled_timesteps) >= self.config.total_timesteps
            )
            self.recorder.write(state.metrics.to_local_dict(), iteration)
            self.recorder.write(train.to_local_dict(), iteration)
            if iteration % self.config.eval_interval == 0 or complete:
                evaluation, state = self.evaluate(state)
                self.recorder.write(
                    add_prefix(evaluation.to_local_dict(), "eval"), iteration
                )
            self.checkpoint_manager.save(iteration, state, force=complete)
        return state

    def close(self):
        try:
            self.workflow.close()
        finally:
            super().close()
