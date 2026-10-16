"""Check EC accounting with different returns on each population shard."""

from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evorl.agent import AgentState
from evorl.distributed import POP_AXIS_NAME
from evorl.evaluators import Evaluator
from evorl.metrics import (
    ECWorkflowMetric,
    MultiObjectiveECWorkflowMetric,
)
from evorl.types import PyTreeDict, State
from evorl.workflows.ec_workflow import (
    DistributedInfo,
    ECWorkflowTemplate,
    MultiObjectiveECWorkflowTemplate,
)
from omegaconf import OmegaConf


class _FitnessEvaluator(Evaluator):
    def evaluate(self, agent_state, key, num_episodes):
        return PyTreeDict(
            episode_returns=agent_state.params + jnp.arange(num_episodes),
            episode_lengths=jnp.full((num_episodes,), 2),
        )


class _FixedPopulation:
    def ask(self, state):
        return state.population, state

    def tell(self, state, fitnesses):
        return PyTreeDict(), state


@pytest.mark.parametrize("multi_objective", [False, True])
@pytest.mark.parametrize("num_devices", [1, 2])
def test_ec_metrics_count_only_local_episodes(num_devices, multi_objective):
    if jax.local_device_count() < num_devices:
        pytest.skip("Run with JAX_NUM_CPU_DEVICES=2 for the multi-device cases")

    base = MultiObjectiveECWorkflowTemplate if multi_objective else ECWorkflowTemplate

    class FitnessWorkflow(base):
        def _replace_actor_params(self, agent_state, params):
            return agent_state.replace(params=params)

    workflow = FitnessWorkflow(
        env=None,
        agent=None,
        ec_optimizer=_FixedPopulation(),
        ec_evaluator=_FitnessEvaluator(
            env=Mock(num_envs=1), action_fn=None, max_episode_steps=2
        ),
        config=OmegaConf.create(
            {
                "checkpoint": {"enable": False},
                "episodes_for_fitness": 2,
                "metric_names": ["episode_returns", "episode_lengths"],
            }
        ),
    )
    mesh = jax.make_mesh(
        (num_devices,),
        (POP_AXIS_NAME,),
        axis_types=(jax.sharding.AxisType.Explicit,),
        devices=jax.local_devices()[:num_devices],
    )
    with jax.set_mesh(mesh):
        workflow.dp_axis_name = POP_AXIS_NAME
        FitnessWorkflow.enable_shmap()
        FitnessWorkflow.enable_jit()
        metrics = (
            MultiObjectiveECWorkflowMetric()
            if multi_objective
            else ECWorkflowMetric(best_objective=jnp.float32(5))
        ).replace(sampled_episodes=jnp.uint32(7))
        state = State(
            key=jax.random.split(jax.random.key(0), num_devices),
            agent_state=AgentState(params=jnp.float32(0)),
            ec_opt_state=PyTreeDict(population=jnp.array([1, 2, 10, 20], jnp.float32)),
            metrics=metrics,
            distributed_info=DistributedInfo(
                rank=jnp.arange(num_devices), world_size=num_devices
            ),
        )
        state = jax.device_put(
            state,
            State(
                key=jax.P(POP_AXIS_NAME),
                agent_state=jax.P(),
                ec_opt_state=jax.P(),
                metrics=jax.P(),
                distributed_info=jax.P(POP_AXIS_NAME),
            ),
        )
        _, updated = workflow.step(state)
        if not multi_objective:
            for replica in updated.metrics.best_objective.addressable_shards:
                assert float(replica.data) == 21
        for replica in updated.metrics.sampled_episodes.addressable_shards:
            assert int(replica.data) == 7 + 4 * 2
        for replica in updated.metrics.sampled_timesteps_m.addressable_shards:
            np.testing.assert_allclose(replica.data, 4 * 2 * 2 / 1e6)
        if not multi_objective:
            _, updated = workflow.step(updated)
            assert int(updated.metrics.sampled_episodes) == 7 + 2 * 4 * 2
            assert float(updated.metrics.best_objective) == 21
    workflow.close()


@pytest.mark.parametrize("mesh_size", [1, 2])
def test_ec_setup_uses_its_own_mesh_size(mesh_size):
    if jax.local_device_count() < 2:
        pytest.skip("Run with JAX_NUM_CPU_DEVICES=2 to check a smaller workflow mesh")

    workflow = object.__new__(ECWorkflowTemplate)
    workflow.dp_axis_name = POP_AXIS_NAME
    workflow.mesh = jax.make_mesh(
        (mesh_size,),
        (POP_AXIS_NAME,),
        axis_types=(jax.sharding.AxisType.Explicit,),
        devices=jax.local_devices()[:mesh_size],
    )
    workflow._setup_agent_and_optimizer = lambda key: (
        AgentState(params=jnp.float32(0)),
        PyTreeDict(),
    )
    with jax.set_mesh(workflow.mesh):
        state = workflow.setup(jax.random.key(0))
    assert state.distributed_info.world_size == mesh_size
    assert state.key.shape == (mesh_size,)
    np.testing.assert_array_equal(state.distributed_info.rank, np.arange(mesh_size))
