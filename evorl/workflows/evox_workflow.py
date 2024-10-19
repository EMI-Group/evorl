from collections.abc import Callable, Sequence
from functools import partial
import numpy as np
import wandb

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from omegaconf import DictConfig
import orbax.checkpoint as ocp

from evox import Algorithm, Problem, State as EvoXState
from evox.workflows import StdWorkflow as EvoXWorkflow
from evox.operators import non_dominated_sort

from evorl.agent import Agent, AgentState
from evorl.algorithms.ec.so.es_base import logger
from evorl.distributed import get_global_ranks, psum, split_key_to_devices, tree_unpmap
from evorl.evaluators import Evaluator
from evorl.metrics import EvaluateMetric, MetricBase
from evorl.recorders import get_1d_array_statistics
from evorl.types import State
from .ec_workflow import ECWorkflow, ECWorkflowMetric, TrainMetric


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


class EvoXESWorkflowTemplate(EvoXWorkflowWrapper):
    def __init__(
        self,
        config: DictConfig,
        agent: Agent,
        evaluator: Evaluator,
        algorithm: Algorithm,
        problem: Problem,
        opt_direction: str | Sequence[str] = "max",
        candidate_transforms: Sequence[Callable] = (),
        fitness_transforms: Sequence[Callable] = (),
    ):
        super().__init__(
            config=config,
            agent=agent,
            algorithm=algorithm,
            problem=problem,
            opt_direction=opt_direction,
            candidate_transforms=candidate_transforms,
            fitness_transforms=fitness_transforms,
        )

        # An extra evaluator for pop_center
        self.evaluator = evaluator

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        num_devices = jax.device_count()

        if config.num_envs % num_devices != 0:
            logger.warning(
                f"num_envs ({config.num_envs}) must be divisible by the number of devices ({num_devices}), "
                f"rescale eval_episodes to {config.eval_episodes // num_devices}"
            )

        config.eval_episodes = config.eval_episodes // num_devices

    def _get_pop_center(self, state: State) -> AgentState:
        raise NotImplementedError

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        """Evaluate the policy with the mean of CMAES"""
        key, eval_key = jax.random.split(state.key, num=2)

        agent_state = self._get_pop_center(state)

        # [#episodes]
        raw_eval_metrics = self.evaluator.evaluate(
            agent_state, eval_key, num_episodes=self.config.eval_episodes
        )

        eval_metrics = EvaluateMetric(
            episode_returns=raw_eval_metrics.episode_returns.mean(),
            episode_lengths=raw_eval_metrics.episode_lengths.mean(),
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return eval_metrics, state.replace(key=key)

    def _record_callback(
        self,
        evox_state: EvoXState,
        iters: int = 0,
    ) -> None:
        """
        Add some customized metrics on evox_state
        """
        pass

    def learn(self, state: State) -> State:
        start_iteration = tree_unpmap(state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            workflow_metrics = tree_unpmap(workflow_metrics, self.pmap_axis_name)
            self.recorder.write(workflow_metrics.to_local_dict(), iters)

            train_metrics = tree_unpmap(train_metrics, self.pmap_axis_name)
            train_metrics_dict = train_metrics.to_local_dict()
            train_metrics_dict = jtu.tree_map(
                partial(get_1d_array_statistics, histogram=True),
                train_metrics.to_local_dict(),
            )
            self.recorder.write(train_metrics_dict, iters)

            if iters % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)
                eval_metrics = tree_unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write(
                    {"eval/pop_center": eval_metrics.to_local_dict()}, iters
                )

            self._record_callback(state.evox_state, iters)

            self.checkpoint_manager.save(
                iters,
                args=ocp.args.StandardSave(
                    tree_unpmap(state, self.pmap_axis_name),
                ),
            )

    @classmethod
    def enable_jit(cls) -> None:
        super().enable_jit()
        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))

    @classmethod
    def enable_pmap(cls, axis_name) -> None:
        super().enable_pmap(axis_name)
        cls.evaluate = jax.pmap(
            cls.evaluate, axis_name, static_broadcasted_argnums=(0,)
        )


class EvoXMOWorkflowTemplate(EvoXWorkflowWrapper):
    def setup(self, key: chex.PRNGKey) -> State:
        state = super().setup(key)
        for metric_name in self.config.obj_names:
            wandb.define_metric(f"pf_objectives.{metric_name}.val", hidden=True)

        return state

    def learn(self, state: State) -> State:
        start_iteration = tree_unpmap(state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            train_metrics = tree_unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = tree_unpmap(workflow_metrics, self.pmap_axis_name)
            self.recorder.write(workflow_metrics.to_local_dict(), iters)

            cpu_device = jax.devices("cpu")[0]
            with jax.default_device(cpu_device):
                objectives = jax.device_put(train_metrics.objectives, cpu_device)
                fitnesses = objectives * self._workflow.opt_direction
                pf_rank = non_dominated_sort(fitnesses, "scan")
                pf_objectives = train_metrics.objectives[pf_rank == 0]

            train_metrics_dict = {}
            metric_names = self.config.obj_names
            objectives = np.asarray(objectives)
            pf_objectives = np.asarray(pf_objectives)
            train_metrics_dict["objectives"] = {
                metric_names[i]: get_1d_array_statistics(
                    objectives[:, i], histogram=True
                )
                for i in range(len(metric_names))
            }

            train_metrics_dict["pf_objectives"] = {
                metric_names[i]: get_1d_array_statistics(
                    pf_objectives[:, i], histogram=True
                )
                for i in range(len(metric_names))
            }
            train_metrics_dict["num_pf"] = pf_objectives.shape[0]

            self.recorder.write(train_metrics_dict, iters)

            self.checkpoint_manager.save(
                iters,
                args=ocp.args.StandardSave(
                    tree_unpmap(state, self.pmap_axis_name),
                ),
            )
