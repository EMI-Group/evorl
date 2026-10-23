from evorl.networks.utils import model_state_axes
import copy
import logging
import math
from omegaconf import DictConfig, OmegaConf
from typing import Self

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.distributed import POP_AXIS_NAME, all_gather
from evorl.metrics import (
    MetricBase,
    ECWorkflowMetric,
    MultiObjectiveECWorkflowMetric,
    ECTrainMetric,
)
from evorl.ec.optimizers import EvoOptimizer, ECState
from evorl.envs import BaseEnv
from evorl.sample_batch import SampleBatch
from evorl.evaluators import Evaluator, EpisodeCollector
from evorl.agent import Agent, AgentState, AgentStateAxis
from evorl.distributed import get_global_ranks, pmax, psum
from evorl.types import State, PyTreeData, pytree_field, Params
from evorl.utils.rl_toolkits import flatten_pop_rollout_episode
from evorl.utils.jax_utils import tree_stop_gradient

from .workflow import Workflow


logger = logging.getLogger(__name__)


class DistributedInfo(PyTreeData):
    """Distributed information for multi-devices training."""

    rank: int = pytree_field(default_factory=lambda: jnp.zeros((), dtype=jnp.int32))
    world_size: int = pytree_field(default=1, static=True)


class ECWorkflow(Workflow):
    """Base Workflow for EC (Evolutionary Computation) algorithms."""

    def __init__(self, config: DictConfig):
        """Initialize the ECWorkflow instance.

        Args:
            config: the config object
        """
        super().__init__(config)

        self.dp_axis_name = None

    @property
    def is_multi_device(self) -> bool:
        """Whether multi-devices training is enabled."""
        return self.dp_axis_name is not None

    @classmethod
    def build_from_config(
        cls,
        config: DictConfig,
        multi_device: bool = False,
        jit: bool = True,
    ):
        """Build the ec workflow instance from the config.

        Call on a fresh subclass for each execution configuration, as in
        `scripts/train.py`. The enable_* class methods transform that subclass.

        Args:
            config: Config of the workflow.
            multi_device: Whether multi-devices training is enabled.
            jit: Whether jit is enabled.
        """
        config = copy.deepcopy(config)  # avoid in-place modification

        devices = jax.local_devices()

        if multi_device:
            OmegaConf.set_readonly(config, False)
            cls._rescale_config(config)

        OmegaConf.set_readonly(config, True)

        workflow = cls._build_from_config(config)
        if multi_device:
            workflow.dp_axis_name = POP_AXIS_NAME
            workflow.mesh = jax.make_mesh(
                (len(devices),),
                (POP_AXIS_NAME,),
                axis_types=(jax.sharding.AxisType.Explicit,),
                devices=devices,
            )
            with jax.set_mesh(workflow.mesh):
                cls.enable_shmap()

        if jit:
            cls.enable_jit()

        return workflow

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        """Customize the process of building the workflow instance from the config.

        Args:
            config: Config of the workflow

        Returns:
            workflow: the created workflow instance
        """
        raise NotImplementedError

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        """Validate the population and split observation warmup across devices."""
        num_devices = jax.local_device_count()
        if config.pop_size < num_devices or config.pop_size % num_devices:
            raise ValueError(
                f"pop_size({config.pop_size}) must be divisible by num_devices({num_devices})"
            )
        if config.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if config.random_timesteps < 0:
            raise ValueError("random_timesteps must be nonnegative")
        if not config.normalize_obs or config.get("normalize_obs_mode") == "RS":
            return
        if config.random_timesteps == 0:
            raise ValueError(
                "random_timesteps must be positive when normalize_obs=True"
            )

        # Each device collects complete batches of num_envs observations.
        global_batches = math.ceil(config.random_timesteps / config.num_envs)
        local_batches = math.ceil(global_batches / num_devices)
        global_samples = local_batches * config.num_envs * num_devices
        single_device_samples = global_batches * config.num_envs
        if global_samples != single_device_samples:
            logger.warning(
                f"random_timesteps({config.random_timesteps}) requires complete "
                f"environment batches; multi-device warmup collects {global_samples} "
                f"instead of {single_device_samples} observations"
            )
        config.random_timesteps = local_batches * config.num_envs

    @classmethod
    def enable_jit(cls) -> None:
        """Define which methods should be jitted.

        By default, the workflow's `step()` method is jitted.
        """
        cls.step = jax.jit(cls.step, static_argnums=(0,))

    @classmethod
    def enable_shmap(cls) -> None:
        """Define distributed method boundaries in the concrete workflow template."""
        raise NotImplementedError


class ECWorkflowTemplate(ECWorkflow):
    """Workflow template for EC algorithms.

    Attributes:
        env: Environment object.
        agent: Workflow-sepecific agent object.
        ec_optimizer: EC Optimizer of the agent.
        ec_evaluator: Evaluator object used in `self.evaluation()`.
        agent_state_vmap_axes: Vmap axis for the agent state.
        config: Config of the workflow.
    """

    def __init__(
        self,
        *,
        env: BaseEnv,
        agent: Agent,
        ec_optimizer: EvoOptimizer,
        ec_evaluator: Evaluator | EpisodeCollector,
        agent_state_vmap_axes: AgentStateAxis = 0,
        config: DictConfig,
    ):
        """Initialize the ECWorkflow instance.

        Args:
            env: Environment object.
            agent: Workflow-sepecific agent object.
            ec_optimizer: EC Optimizer of the agent.
            ec_evaluator: Evaluator object used in `self.evaluation()`.
            agent_state_vmap_axes: Vmap axis for the agent state.
            config: Config of the workflow.
        """
        super().__init__(config)

        self.agent = agent
        self.env = env
        self.ec_optimizer = ec_optimizer
        self.ec_evaluator = ec_evaluator
        self.agent_state_vmap_axes = agent_state_vmap_axes

    @classmethod
    def enable_shmap(cls) -> None:
        """Map local computations over workers, keeping self static."""
        state_specs = State(
            key=jax.P(POP_AXIS_NAME),
            agent_state=jax.P(),
            ec_opt_state=jax.P(),
            metrics=jax.P(),
            distributed_info=jax.P(POP_AXIS_NAME),
        )

        original_step = cls.step

        def sharded_step(self, state):
            state = state.replace(
                key=state.key[0],
                distributed_info=jax.tree.map(lambda x: x[0], state.distributed_info),
            )
            metrics, state = original_step(self, state)
            return metrics, state.replace(
                key=state.key[None],
                distributed_info=jax.tree.map(
                    lambda x: x[None], state.distributed_info
                ),
            )

        cls.step = jax.shard_map(
            sharded_step,
            in_specs=(None, state_specs),
            out_specs=(jax.P(), state_specs),
        )

        original_postsetup = cls._postsetup

        def sharded_postsetup(self, state):
            state = state.replace(
                key=state.key[0],
                distributed_info=jax.tree.map(lambda x: x[0], state.distributed_info),
            )
            state = original_postsetup(self, state)
            return state.replace(
                key=state.key[None],
                distributed_info=jax.tree.map(
                    lambda x: x[None], state.distributed_info
                ),
            )

        cls._postsetup = jax.shard_map(
            sharded_postsetup,
            in_specs=(None, state_specs),
            out_specs=state_specs,
        )

    def _setup_agent_and_optimizer(self, key: jax.Array) -> tuple[AgentState, ECState]:
        """Setup Agent and ECOptimizer states.

        Args:
            key: JAX PRNGKey

        Returns:
            Tuple of (agent_state, ec_state)
        """
        raise NotImplementedError

    def _setup_workflow_metrics(self) -> MetricBase:
        """Define Workflow metrics."""
        return ECWorkflowMetric(best_objective=jnp.finfo(jnp.float32).min)

    def setup(self, key: jax.Array) -> State:
        key, agent_key = jax.random.split(key, 2)

        # agent_state: store params not optimized by EC (eg: obs_preprocessor_state)
        agent_state, ec_opt_state = self._setup_agent_and_optimizer(agent_key)
        workflow_metrics = self._setup_workflow_metrics()
        distributed_info = DistributedInfo()

        if self.is_multi_device:
            agent_state, ec_opt_state, workflow_metrics = jax.device_put(
                (agent_state, ec_opt_state, workflow_metrics), jax.P()
            )
            num_workers = self.mesh.shape[self.dp_axis_name]
            key = jax.device_put(
                jax.random.split(key, num_workers), jax.P(self.dp_axis_name)
            )

            distributed_info = DistributedInfo(
                rank=get_global_ranks(self.dp_axis_name),
                world_size=num_workers,
            )

        state = State(
            key=key,
            agent_state=agent_state,
            ec_opt_state=ec_opt_state,
            metrics=workflow_metrics,
            distributed_info=distributed_info,
        )

        state = self._postsetup(state)

        return state

    def _postsetup(self, state: State) -> State:
        """Post-setup state before training.

        By default, no post-setup is applied
        """
        return state

    def _replace_actor_params(
        self, agent_state: AgentState, params: Params
    ) -> AgentState:
        """Define how to replace the pop agent_state from the population params.

        Args:
            agent_state: State of the agent.
            params: Population params.

        Returns:
            New agent_state with replaced population params.
        """
        raise NotImplementedError

    def _update_obs_preprocessor(
        self, agent_state: AgentState, trajectory: SampleBatch
    ) -> AgentState:
        """Update the obs_preprocessor_state based on sampled trajectories.

        By default, don't update obs_preprocessor_state.

        Args:
            agent_state: State of the agent
            trajectory: Episodic trajectory (T, B, ...)
        """
        return agent_state

    def _metrics_to_fitnesses(self, metrics: MetricBase) -> chex.ArrayTree:
        """Convert the rollout metrics to fitnesses.

        By default, use the mean of episode_returns over multiple episodes as fitnesses.

        Args:
            metrics: Rollout metrics.

        Returns:
            Fitnesses of the population.
        """
        return jnp.mean(metrics.episode_returns, axis=-1)

    def step(self, state: State) -> tuple[MetricBase, State]:
        agent_state = state.agent_state
        key, rollout_key = jax.random.split(state.key, 2)

        pop, ec_opt_state = self.ec_optimizer.ask(state.ec_opt_state)
        pop_size = jtu.tree_leaves(pop)[0].shape[0]

        slice_size = pop_size // state.distributed_info.world_size
        eval_pop = jtu.tree_map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, state.distributed_info.rank * slice_size, slice_size, axis=0
            ),
            pop,
        )

        pop_agent_state = self._replace_actor_params(agent_state, eval_pop)
        agent_state_axes = self.agent_state_vmap_axes
        if isinstance(agent_state_axes, AgentState) and not (
            agent_state_axes.model_state is None
            or isinstance(agent_state_axes.model_state, int)
        ):
            agent_state_axes = agent_state_axes.replace(
                model_state=model_state_axes(
                    pop_agent_state.model_state, agent_state_axes.model_state
                )
            )

        if isinstance(self.ec_evaluator, EpisodeCollector):
            # trajectory: [#pop, T, #episodes]
            rollout_metrics, trajectory = jax.vmap(
                self.ec_evaluator.rollout,
                in_axes=(agent_state_axes, 0, None),
            )(
                pop_agent_state,
                jax.random.split(rollout_key, num=slice_size),
                self.config.episodes_for_fitness,
            )
            # [#pop, T, B, ...] -> [T, #pop*B, ...]
            trajectory = flatten_pop_rollout_episode(trajectory)
            trajectory = tree_stop_gradient(trajectory)
            agent_state = self._update_obs_preprocessor(agent_state, trajectory)

        elif isinstance(self.ec_evaluator, Evaluator):
            rollout_metrics = jax.vmap(
                self.ec_evaluator.evaluate,
                in_axes=(agent_state_axes, 0, None),
            )(
                pop_agent_state,
                jax.random.split(rollout_key, num=slice_size),
                self.config.episodes_for_fitness,
            )

        fitnesses = self._metrics_to_fitnesses(rollout_metrics)
        fitnesses = all_gather(
            fitnesses, self.dp_axis_name, axis=0, tiled=True, to="invarying"
        )

        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        sampled_episodes = psum(
            jnp.uint32(slice_size * self.config.episodes_for_fitness),
            self.dp_axis_name,
        )
        sampled_timesteps_m = (
            psum(rollout_metrics.episode_lengths.sum(), self.dp_axis_name) / 1e6
        )

        workflow_metrics = state.metrics.replace(
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
            sampled_timesteps_m=state.metrics.sampled_timesteps_m + sampled_timesteps_m,
            iterations=state.metrics.iterations + 1,
            best_objective=jnp.maximum(
                state.metrics.best_objective,
                pmax(jnp.max(rollout_metrics.episode_returns), self.dp_axis_name),
            ),
        )

        train_metrics = ECTrainMetric(
            objectives=fitnesses,
            ec_metrics=ec_metrics,
        )

        return train_metrics, state.replace(
            key=key,
            agent_state=agent_state,
            ec_opt_state=ec_opt_state,
            metrics=workflow_metrics,
        )

    @classmethod
    def enable_jit(cls) -> None:
        super().enable_jit()
        cls._postsetup = jax.jit(cls._postsetup, static_argnums=(0,))


class MultiObjectiveECWorkflowTemplate(ECWorkflowTemplate):
    """Workflow template for multi-objective EC algorithms."""

    def _metrics_to_fitnesses(self, metrics: MetricBase) -> chex.ArrayTree:
        # [..., episodes, objectives] -> [..., objectives]. Keep one-objective
        # vectors distinct from ordinary scalar fitnesses.
        return jnp.mean(metrics.episode_returns, axis=-2)

    def _setup_workflow_metrics(self) -> MetricBase:
        return MultiObjectiveECWorkflowMetric()

    def step(self, state: State) -> tuple[MetricBase, State]:
        agent_state = state.agent_state
        key, rollout_key = jax.random.split(state.key, 2)

        pop, ec_opt_state = self.ec_optimizer.ask(state.ec_opt_state)
        pop_size = jtu.tree_leaves(pop)[0].shape[0]

        slice_size = pop_size // state.distributed_info.world_size
        eval_pop = jtu.tree_map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, state.distributed_info.rank * slice_size, slice_size, axis=0
            ),
            pop,
        )

        pop_agent_state = self._replace_actor_params(agent_state, eval_pop)
        agent_state_axes = self.agent_state_vmap_axes
        if isinstance(agent_state_axes, AgentState) and not (
            agent_state_axes.model_state is None
            or isinstance(agent_state_axes.model_state, int)
        ):
            agent_state_axes = agent_state_axes.replace(
                model_state=model_state_axes(
                    pop_agent_state.model_state, agent_state_axes.model_state
                )
            )

        if isinstance(self.ec_evaluator, EpisodeCollector):
            # trajectory: [#pop, T, #episodes]
            rollout_metrics, trajectory = jax.vmap(
                self.ec_evaluator.rollout,
                in_axes=(agent_state_axes, 0, None),
            )(
                pop_agent_state,
                jax.random.split(rollout_key, num=slice_size),
                self.config.episodes_for_fitness,
            )
            # [#pop, T, B, ...] -> [T, #pop*B, ...]
            trajectory = flatten_pop_rollout_episode(trajectory)
            trajectory = tree_stop_gradient(trajectory)
            agent_state = self._update_obs_preprocessor(agent_state, trajectory)

        elif isinstance(self.ec_evaluator, Evaluator):
            rollout_metrics = jax.vmap(
                self.ec_evaluator.evaluate,
                in_axes=(agent_state_axes, 0, None),
            )(
                pop_agent_state,
                jax.random.split(rollout_key, num=slice_size),
                self.config.episodes_for_fitness,
            )

        fitnesses = self._metrics_to_fitnesses(rollout_metrics)
        fitnesses = all_gather(
            fitnesses, self.dp_axis_name, axis=0, tiled=True, to="invarying"
        )

        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        sampled_episodes = psum(
            jnp.uint32(slice_size * self.config.episodes_for_fitness),
            self.dp_axis_name,
        )
        sampled_timesteps_m = (
            psum(rollout_metrics.episode_lengths.sum(), self.dp_axis_name) / 1e6
        )

        workflow_metrics = state.metrics.replace(
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
            sampled_timesteps_m=state.metrics.sampled_timesteps_m + sampled_timesteps_m,
            iterations=state.metrics.iterations + 1,
        )

        train_metrics = ECTrainMetric(objectives=fitnesses, ec_metrics=ec_metrics)

        return train_metrics, state.replace(
            key=key,
            agent_state=agent_state,
            ec_opt_state=ec_opt_state,
            metrics=workflow_metrics,
        )
