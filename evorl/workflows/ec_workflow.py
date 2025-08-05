import copy
import logging
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Self

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
from evorl.envs import Env
from evorl.sample_batch import SampleBatch
from evorl.evaluators import Evaluator, EpisodeCollector
from evorl.agent import Agent, AgentState, AgentStateAxis
from evorl.distributed import get_global_ranks, psum, split_key_to_devices
from evorl.types import State, PyTreeData, pytree_field, Params, PyTreeDict
from evorl.utils.rl_toolkits import flatten_pop_rollout_episode
from evorl.utils.jax_utils import tree_stop_gradient

from .workflow import Workflow


logger = logging.getLogger(__name__)


class DistributedInfo(PyTreeData):
    """Distributed information for multi-devices training."""

    rank: int = jnp.zeros((), dtype=jnp.int32)
    world_size: int = pytree_field(default=1, static=True)


class ECWorkflow(Workflow):
    """Base Workflow for EC (Evolutionary Computation) algorithms."""

    def __init__(self, config: DictConfig):
        """Initialize the ECWorkflow instance.

        Args:
            config: the config object
        """
        super().__init__(config)

        self.pmap_axis_name = None
        self.devices = jax.local_devices()[:1]

    @property
    def enable_multi_devices(self) -> bool:
        """Whether multi-devices training is enabled."""
        return self.pmap_axis_name is not None

    @classmethod
    def build_from_config(
        cls,
        config: DictConfig,
        enable_multi_devices: bool = False,
        enable_jit: bool = True,
    ):
        """Build the ec workflow instance from the config.

        Args:
            config: Config of the workflow
            enable_multi_devices: Whether multi-devices training is enabled
            enable_jit: Whether jit is enabled
        """
        config = copy.deepcopy(config)  # avoid in-place modification

        devices = jax.local_devices()

        if enable_multi_devices:
            cls.enable_pmap(POP_AXIS_NAME)
            OmegaConf.set_readonly(config, False)
            cls._rescale_config(config)
        elif enable_jit:
            cls.enable_jit()

        OmegaConf.set_readonly(config, True)

        workflow = cls._build_from_config(config)
        if enable_multi_devices:
            workflow.pmap_axis_name = POP_AXIS_NAME
            workflow.devices = devices

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
        """Customize the logic of rescaling the config settings when multi-devices training is enabled.

        When enable_multi_devices=True, rescale config settings in-place to match multi-devices

        Args:
            config: Config of the workflow
        """
        pass

    @classmethod
    def enable_jit(cls) -> None:
        """Define which methods should be jitted.

        By default, the workflow's `step()` method is jitted.
        """
        cls.step = jax.jit(cls.step, static_argnums=(0,))

    @classmethod
    def enable_pmap(cls, axis_name) -> None:
        """Define which methods should be pmaped.

        This method defines the multi-device behavior. By default, the workflow's `step()` method is pmaped.
        """
        cls.step = jax.pmap(cls.step, axis_name, static_broadcasted_argnums=(0,))


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
        env: Env,
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

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, ECState]:
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

    def setup(self, key: chex.PRNGKey) -> State:
        key, agent_key = jax.random.split(key, 2)

        # agent_state: store params not optimized by EC (eg: obs_preprocessor_state)
        agent_state, ec_opt_state = self._setup_agent_and_optimizer(agent_key)
        workflow_metrics = self._setup_workflow_metrics()
        distributed_info = DistributedInfo()

        if self.enable_multi_devices:
            agent_state, ec_opt_state, workflow_metrics = jax.device_put_replicated(
                (agent_state, ec_opt_state, workflow_metrics), self.devices
            )
            key = split_key_to_devices(key, self.devices)

            distributed_info = DistributedInfo(
                rank=get_global_ranks(),
                world_size=jax.device_count(),
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

        if isinstance(self.ec_evaluator, EpisodeCollector):
            # trajectory: [#pop, T, #episodes]
            rollout_metrics, trajectory = jax.vmap(
                self.ec_evaluator.rollout,
                in_axes=(self.agent_state_vmap_axes, 0, None),
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
                in_axes=(self.agent_state_vmap_axes, 0, None),
            )(
                pop_agent_state,
                jax.random.split(rollout_key, num=slice_size),
                self.config.episodes_for_fitness,
            )

        fitnesses = self._metrics_to_fitnesses(rollout_metrics)
        fitnesses = all_gather(fitnesses, self.pmap_axis_name, axis=0, tiled=True)

        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        sampled_episodes = psum(
            jnp.uint32(pop_size * self.config.episodes_for_fitness),
            self.pmap_axis_name,
        )
        sampled_timesteps_m = (
            psum(rollout_metrics.episode_lengths.sum(), self.pmap_axis_name) / 1e6
        )

        workflow_metrics = state.metrics.replace(
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
            sampled_timesteps_m=state.metrics.sampled_timesteps_m + sampled_timesteps_m,
            iterations=state.metrics.iterations + 1,
            best_objective=jnp.maximum(
                state.metrics.best_objective, jnp.max(rollout_metrics.episode_returns)
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

    @classmethod
    def enable_pmap(cls, axis_name) -> None:
        super().enable_pmap(axis_name)
        cls._postsetup = jax.pmap(
            cls._postsetup, axis_name, static_broadcasted_argnums=(0,)
        )


class MultiObjectiveECWorkflowTemplate(ECWorkflowTemplate):
    """Workflow template for multi-objective EC algorithms."""

    def _metrics_to_fitnesses(self, metrics: MetricBase) -> chex.ArrayTree:
        fitnesses = PyTreeDict(
            {k: jnp.mean(metrics[k], axis=-1) for k in self.config.metric_names}
        )
        fitnesses = jnp.stack(list(fitnesses.values()), axis=-1)
        if fitnesses.shape[-1] == 1:
            fitnesses = fitnesses.squeeze(-1)

        return fitnesses

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

        if isinstance(self.ec_evaluator, EpisodeCollector):
            # trajectory: [#pop, T, #episodes]
            rollout_metrics, trajectory = jax.vmap(
                self.ec_evaluator.rollout,
                in_axes=(self.agent_state_vmap_axes, 0, None),
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
                in_axes=(self.agent_state_vmap_axes, 0, None),
            )(
                pop_agent_state,
                jax.random.split(rollout_key, num=slice_size),
                self.config.episodes_for_fitness,
            )

        fitnesses = self._metrics_to_fitnesses(rollout_metrics)
        fitnesses = all_gather(fitnesses, self.pmap_axis_name, axis=0, tiled=True)

        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        sampled_episodes = psum(
            jnp.uint32(pop_size * self.config.episodes_for_fitness),
            self.pmap_axis_name,
        )
        sampled_timesteps_m = (
            psum(rollout_metrics.episode_lengths.sum(), self.pmap_axis_name) / 1e6
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
