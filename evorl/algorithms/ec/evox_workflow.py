from collections.abc import Callable, Sequence
from omegaconf import DictConfig


import chex
import jax
import jax.numpy as jnp
from evox import Algorithm, Problem
from evox.workflows import StdWorkflow as EvoXWorkflow


from evorl.agent import Agent
from evorl.distributed import get_global_ranks, psum, split_key_to_devices
from evorl.metrics import MetricBase
from evorl.types import State
from evorl.workflows.ec_workflow import ECWorkflow, ECWorkflowMetric, TrainMetric


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
            best_objective=jnp.full(obj_shape, jnp.finfo(jnp.float32).min)
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

    def step(self, state: State) -> tuple[TrainMetric, State]:
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

    @classmethod
    def enable_jit(cls) -> None:
        cls.step = jax.jit(cls.step, static_argnums=(0,))

    @classmethod
    def enable_pmap(cls, axis_name) -> None:
        cls.step = jax.pmap(cls.step, axis_name, static_broadcasted_argnums=(0,))
