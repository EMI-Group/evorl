import copy
import logging
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.distributed import POP_AXIS_NAME, all_gather
from evorl.metrics import MetricBase
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


class ECWorkflowMetric(MetricBase):
    best_objective: chex.Array
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_timesteps_m: chex.Array = jnp.zeros((), dtype=jnp.float32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class MultiObjectiveECWorkflowMetric(MetricBase):
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_timesteps_m: chex.Array = jnp.zeros((), dtype=jnp.float32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class TrainMetric(MetricBase):
    objectives: chex.Array
    ec_metrics: chex.ArrayTree


class DistributedInfo(PyTreeData):
    rank: int = jnp.zeros((), dtype=jnp.int32)
    world_size: int = pytree_field(default=1, static=True)


class ECWorkflow(Workflow):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.pmap_axis_name = None
        self.devices = jax.local_devices()[:1]

    @property
    def enable_multi_devices(self) -> bool:
        return self.pmap_axis_name is not None

    @classmethod
    def build_from_config(
        cls,
        config: DictConfig,
        enable_multi_devices: bool = False,
        enable_jit: bool = True,
    ):
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
        raise NotImplementedError

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        """
        When enable_multi_devices=True, rescale config settings in-place to match multi-devices.
        Note: not need for EvoX part, as it's already handled by EvoX.
        """
        pass

    @classmethod
    def enable_jit(cls) -> None:
        cls.step = jax.jit(cls.step, static_argnums=(0,))

    @classmethod
    def enable_pmap(cls, axis_name) -> None:
        cls.step = jax.pmap(cls.step, axis_name, static_broadcasted_argnums=(0,))


class ECWorkflowTemplate(ECWorkflow):
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
        super().__init__(config)

        self.agent = agent
        self.env = env
        self.ec_optimizer = ec_optimizer
        self.ec_evaluator = ec_evaluator
        self.agent_state_vmap_axes = agent_state_vmap_axes

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        num_devices = jax.device_count()

        # Note: in some model, the generated number of individuals may not be pop_size,
        # then adjust accordingly
        if config.pop_size % num_devices != 0:
            new_pop_size = (config.pop_size // num_devices) * num_devices
            logger.warning(
                f"When enable_multi_devices=True, pop_size ({config.pop_size}) should be divisible by num_devices ({num_devices}), set new pop_size to {new_pop_size}"
            )

            config.pop_size = new_pop_size

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, ECState]:
        # agent_key, ec_key = jax.random.split(key, 2)
        # agent_state = self.agent.init(self.env.obs_space, self.env.action_space, agent_key)
        raise NotImplementedError

    def _setup_workflow_metrics(self) -> MetricBase:
        """
        Customize the workflow metrics.
        """

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
        return state

    def _replace_actor_params(
        self, agent_state: AgentState, params: Params
    ) -> AgentState:
        raise NotImplementedError

    def _update_obs_preprocessor(
        self, agent_state: AgentState, trajectory: SampleBatch
    ) -> AgentState:
        """
        By default, don't update obs_preprocessor_state.

        Args:
            agent_state: agent_state
            trajectory: Episodic trajectory (T, B, ...)
        """
        return agent_state

    def _metrics_to_fitnesses(self, metrics: MetricBase) -> chex.ArrayTree:
        return jnp.mean(metrics.episode_returns, axis=-1)

    def step(self, state: State) -> tuple[MetricBase, State]:
        agent_state = state.agent_state
        key, rollout_key = jax.random.split(state.key, 2)

        pop, ec_opt_state = self.ec_optimizer.ask(state.ec_opt_state)
        pop_size = jax.tree_leaves(pop)[0].shape[0]

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

        train_metrics = TrainMetric(
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
        pop_size = jax.tree_leaves(pop)[0].shape[0]

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

        train_metrics = TrainMetric(objectives=fitnesses)

        return train_metrics, state.replace(
            key=key,
            agent_state=agent_state,
            ec_opt_state=ec_opt_state,
            metrics=workflow_metrics,
        )
