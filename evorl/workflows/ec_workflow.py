import copy
import logging
from collections.abc import Callable, Sequence
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from evox import Algorithm, Problem
from evox.workflows import StdWorkflow as EvoXWorkflow

from evorl.distributed import POP_AXIS_NAME, all_gather
from evorl.metrics import MetricBase
from evorl.ec.optimizers import EvoOptimizer, ECState
from evorl.envs import Env
from evorl.evaluators import Evaluator
from evorl.agent import Agent, AgentState
from evorl.distributed import get_global_ranks, psum, split_key_to_devices
from evorl.types import State, PyTreeData, pytree_field, Params

from .workflow import Workflow

logger = logging.getLogger(__name__)


class ECWorkflowMetric(MetricBase):
    best_objective: chex.Array
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_timesteps_m: chex.Array = jnp.zeros((), dtype=jnp.float32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class TrainMetric(MetricBase):
    objectives: chex.Array


class DistributedInfo(PyTreeData):
    rank: int = jnp.zeros((), dtype=jnp.int32)
    world_size: int = pytree_field(default=1, pytree_node=False)


class ECWorkflow(Workflow):
    def __init__(self, config: DictConfig):
        """
        config:
        devices: a single device or a list of devices.
        """
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

    @staticmethod
    def _rescale_config(config: DictConfig) -> None:
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
        config: DictConfig,
        env: Env,
        agent: Agent,
        ec_optimizer: EvoOptimizer,
        ec_evaluator: Evaluator,
    ):
        super().__init__(config)

        self.agent = agent
        self.env = env
        self.ec_optimizer = ec_optimizer
        self.ec_evaluator = ec_evaluator

    @staticmethod
    def _rescale_config(config: DictConfig) -> None:
        num_devices = jax.device_count()

        if config.pop_size % num_devices != 0:
            logging.warning(
                f"When enable_multi_devices=True, pop_size ({config.pop_size}) should be divisible by num_devices ({num_devices}),"
            )

        config.pop_size = (config.pop_size // num_devices) * num_devices

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

        agent_state, ec_opt_state = self._setup_agent_and_optimizer(agent_key)
        workflow_metrics = self._setup_workflow_metrics()
        distributed_info = DistributedInfo()

        if self.enable_multi_devices:
            ec_opt_state, workflow_metrics = jax.device_put_replicated(
                (ec_opt_state, workflow_metrics), self.devices
            )
            key = split_key_to_devices(key, self.devices)

            distributed_info = DistributedInfo(
                rank=get_global_ranks(),
                world_size=jax.device_count(),
            )

        return State(
            key=key,
            agent_state=agent_state,
            ec_opt_state=ec_opt_state,
            metrics=workflow_metrics,
            distributed_info=distributed_info,
        )

    def _replace_actor_params(
        self, agent_state: AgentState, params: Params
    ) -> AgentState:
        raise NotImplementedError

    def step(self, state: State) -> tuple[MetricBase, State]:
        pop_size = self.config.pop_size

        key, rollout_key, ec_key = jax.random.split(state.key, 3)

        pop = self.ec_optimizer.ask(state.ec_opt_state, ec_key)

        slice_size = pop_size // state.distributed_info.world_size
        eval_pop = jtu.tree_map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, state.distributed_info.rank * slice_size, slice_size, axis=0
            ),
            pop,
        )

        pop_agent_state = self._replace_actor_params(state.agent_state, eval_pop)

        rollout_metrics = self.ec_evaluator.evaluate(
            pop_agent_state,
            jax.random.split(rollout_key, num=slice_size),
            num_episodes=self.config.episodes_for_fitness,
        )
        fitnesses = jnp.mean(rollout_metrics.episode_returns, axis=-1)
        fitnesses = all_gather(fitnesses, self.pmap_axis_name, axis=0, tiled=True)

        ec_opt_state = self.ec_optimizer.tell(state.ec_opt_state, pop, fitnesses)

        sampled_episodes = psum(
            jnp.uint32(self.config.pop_size * self.config.episodes_for_fitness),
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

        train_metrics = TrainMetric(objectives=fitnesses)

        return train_metrics, state.replace(
            key=key,
            ec_opt_state=ec_opt_state,
            metrics=workflow_metrics,
        )


class EvoXWorkflowWrapper(ECWorkflow):
    def __init__(
        self,
        config: DictConfig,
        agent: Agent,
        algorithm: Algorithm,
        problem: Problem,
        opt_direction: str | Sequence[str] = "max",
        candidate_transforms: Sequence[Callable] = (),
        fitness_transforms: Sequence[Callable] = (),
    ):
        super().__init__(config)

        self.agent = agent
        self._workflow = EvoXWorkflow(
            algorithm=algorithm,
            problem=problem,
            monitors=[],
            opt_direction=opt_direction,
            candidate_transforms=candidate_transforms,
            fitness_transforms=fitness_transforms,
            jit_step=False,  # don't jit internally
        )
        self.pmap_axis_name = None
        self.devices = jax.local_devices()[:1]

    def _setup_workflow_metrics(self) -> MetricBase:
        """
        Customize the workflow metrics.
        """
        if self._workflow.problem.num_objectives == 1:
            obj_shape = ()
        elif self._workflow.problem.num_objectives > 1:
            obj_shape = (self._workflow.problem.num_objectives,)
        else:
            raise ValueError("Invalid num_objectives")

        return ECWorkflowMetric(
            best_objective=jnp.full(obj_shape, jnp.finfo(jnp.float32).max)
            * self._workflow.opt_direction
        )

    def setup(self, key: chex.PRNGKey) -> State:
        key, evox_key = jax.random.split(key, 2)
        evox_state = self._workflow.init(evox_key)
        workflow_metrics = self._setup_workflow_metrics()

        if self.enable_multi_devices:
            # Note: we don't use evox's enable_multi_devices(),
            # instead we use our own implementation
            self._workflow.pmap_axis_name = self.pmap_axis_name
            self._workflow.devices = self.devices

            evox_state, workflow_metrics = jax.device_put_replicated(
                (evox_state, workflow_metrics), self.devices
            )
            key = split_key_to_devices(key, self.devices)
            evox_state = evox_state.replace(
                rank=get_global_ranks(), world_size=jax.device_count()
            )

        return State(key=key, evox_state=evox_state, metrics=workflow_metrics)

    def step(self, state: State) -> tuple[MetricBase, State]:
        opt_direction = self._workflow.opt_direction

        train_info, evox_state = self._workflow.step(state.evox_state)

        problem_state = state.evox_state.get_child_state("problem")
        sampled_episodes = psum(problem_state.sampled_episodes, self.pmap_axis_name)
        sampled_timesteps_m = (
            psum(problem_state.sampled_timesteps, self.pmap_axis_name) / 1e6
        )
        # turn back to the original objectives
        # Note: train_info['fitness'] is already all-gathered in evox
        fitnesses = train_info["fitness"]

        train_metrics = TrainMetric(objectives=fitnesses * opt_direction)

        workflow_metrics = state.metrics.replace(
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
            sampled_timesteps_m=state.metrics.sampled_timesteps_m + sampled_timesteps_m,
            iterations=state.metrics.iterations + 1,
            best_objective=jnp.minimum(
                state.metrics.best_objective * opt_direction, jnp.min(fitnesses, axis=0)
            )
            * opt_direction,
        )

        state = state.replace(evox_state=evox_state, metrics=workflow_metrics)

        return train_metrics, state
