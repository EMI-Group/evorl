"""Workflow summaries reduce episodes and devices while keeping reward axes."""

from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.agent import AgentState
from evorl.algorithms.meta.pbt_workflow import PBTWorkflowTemplate
from evorl.algorithms.random_agent import RandomAgentWorkflow
from evorl.distributed import META_POP_AXIS_NAME
from evorl.metrics import EvaluateMetric
from evorl.types import State
from evorl.workflows import OnPolicyWorkflow, OffPolicyWorkflow


class _FixedEvaluator:
    """Distinct episode, objective, member and device values expose wrong axes."""

    def __init__(self, rewards):
        self.rewards = rewards

    def evaluate(self, agent_state, key, num_episodes):
        offset = agent_state.model_state
        axes = jax.sharding.get_abstract_mesh().manual_axes
        rank = jax.lax.axis_index("DP") if "DP" in axes else 0
        return EvaluateMetric(
            episode_return=jax.tree.map(lambda x: x + offset + 10 * rank, self.rewards),
            episode_length=jnp.array([2.0, 4.0]) + 2 * rank,
        )


@pytest.mark.parametrize("kind", ["onpolicy", "offpolicy", "random"])
@pytest.mark.parametrize(
    "objectives", [None, 1, 3], ids=["scalar", "vector_one", "vector_three"]
)
@pytest.mark.parametrize("distributed", [False, True])
def test_evaluation_keeps_objectives_and_nested_population(
    kind, objectives, distributed, mesh_factory, jit
):
    rewards = jnp.array([1.0, 3.0])
    if objectives is not None:
        rewards = rewards[:, None] * jnp.arange(1, objectives + 1)
    evaluator = _FixedEvaluator(rewards)
    config = OmegaConf.create(dict(checkpoint=dict(enable=False), eval_episodes=2))
    kwargs = dict(
        env=None, agent=None, optimizer=None, evaluator=evaluator, config=config
    )
    if kind == "random":
        workflow = RandomAgentWorkflow(agent=None, evaluator=evaluator, config=config)
    elif kind == "offpolicy":
        workflow = OffPolicyWorkflow(**kwargs, replay_buffer=None)
    else:
        workflow = OnPolicyWorkflow(**kwargs)

    mesh = mesh_factory() if distributed else None
    try:
        with jax.set_mesh(mesh) if distributed else nullcontext():
            evaluate = workflow.evaluate
            if distributed:
                workflow.dp_axis_name = "DP"
                evaluate = jax.shard_map(evaluate, in_specs=jax.P(), out_specs=jax.P())
            for shape in ((), (2, 3)):
                offsets = jnp.arange(int(np.prod(shape)), dtype=jnp.float32).reshape(
                    shape
                )
                state = State(
                    key=jax.random.split(jax.random.key(1), shape),
                    agent_state=AgentState(model_state=offsets),
                )
                fn = evaluate if not shape else jax.vmap(jax.vmap(evaluate))
                result, updated = (jax.jit(fn) if jit else fn)(state)
                rank_mean = (mesh.size - 1) / 2 if distributed else 0

                for got, source in zip(
                    jax.tree.leaves(result.episode_return),
                    jax.tree.leaves(rewards),
                    strict=True,
                ):
                    average = np.asarray(source).mean(axis=0)
                    assert got.shape == (*shape, *average.shape)
                    offset = np.asarray(offsets).reshape(
                        (*shape, *((1,) * average.ndim))
                    )
                    np.testing.assert_allclose(got, average + offset + 10 * rank_mean)
                np.testing.assert_allclose(result.episode_length, 3 + 2 * rank_mean)
                assert result.episode_length.shape == shape
                np.testing.assert_array_equal(updated.agent_state.model_state, offsets)
    finally:
        workflow.close()


@pytest.mark.parametrize("objectives", [None, 1, 3])
@pytest.mark.parametrize("num_devices", [1, 2])
def test_pbt_evaluation_keeps_each_members_reward_coordinates(
    objectives, num_devices, mesh_factory, jit
):
    rewards = jnp.array([1.0, 3.0])
    if objectives is not None:
        rewards = rewards[:, None] * jnp.arange(1, objectives + 1)
    evaluator = _FixedEvaluator(rewards)
    config = OmegaConf.create(
        dict(checkpoint=dict(enable=False), eval_episodes=2, pop_size=4)
    )
    # This checks the generic evaluation path, without invoking PBT selection.
    workflow = PBTWorkflowTemplate(None, evaluator, config)
    workflow.mesh = mesh_factory(shape=(num_devices,), names=(META_POP_AXIS_NAME,))
    try:
        with jax.set_mesh(workflow.mesh):
            state = jax.device_put(
                State(
                    key=jax.random.key(2),
                    pop_workflow_state=State(
                        agent_state=AgentState(model_state=jnp.arange(4) * 100)
                    ),
                ),
                State(key=jax.P(), pop_workflow_state=jax.P(META_POP_AXIS_NAME)),
            )
            evaluate = jax.jit(workflow.evaluate) if jit else workflow.evaluate
            result, _ = evaluate(state)
        expected = np.asarray(rewards).mean(0)
        offsets = (np.arange(4) * 100).reshape(4, *((1,) * expected.ndim))
        assert result.pop_episode_returns.shape == (4, *expected.shape)
        np.testing.assert_allclose(result.pop_episode_returns, expected + offsets)
        np.testing.assert_allclose(result.pop_episode_lengths, 3)
        assert result.pop_episode_lengths.shape == (4,)
    finally:
        workflow.close()
