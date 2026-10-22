"""Check that ES persists its final state outside the save interval."""

from functools import partial
from unittest.mock import Mock

import chex
import optax
from flax import nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
from omegaconf import OmegaConf

from evorl.algorithms.ppo import PPOAgent
from evorl.algorithms.td3 import TD3Agent
from evorl.utils.orbax_utils import load, save, to_shape_dtype_struct
from evorl.algorithms.ec.so.es_workflow import ESWorkflowTemplate
from evorl.metrics import ECTrainMetric, ECWorkflowMetric, EvaluateMetric
from evorl.types import PyTreeDict, State
from evorl.utils.orbax_utils import CheckpointManager

from tests.helpers.envs import ToyEnv
from tests.helpers.agents import build_agent


class _CountingESWorkflow(ESWorkflowTemplate):
    def step(self, state):
        return ECTrainMetric(
            objectives=jnp.ones(2), ec_metrics=PyTreeDict()
        ), state.replace(
            metrics=state.metrics.replace(iterations=state.metrics.iterations + 1)
        )

    def evaluate(self, state):
        return EvaluateMetric(
            episode_returns=jnp.float32(0), episode_lengths=jnp.float32(0)
        ), state


@pytest.mark.parametrize("start_iteration", [0, 1])
def test_es_saves_final_checkpoint_between_save_intervals(tmp_path, start_iteration):
    workflow = object.__new__(_CountingESWorkflow)
    workflow.config = OmegaConf.create({"num_iters": 3, "eval_interval": 10})
    workflow.recorder = Mock()
    mesh = jax.make_mesh(
        (1,),
        ("devices",),
        axis_types=(jax.sharding.AxisType.Explicit,),
        devices=jax.local_devices()[:1],
    )
    with jax.set_mesh(mesh):
        state = State(
            metrics=ECWorkflowMetric(
                best_objective=jnp.float32(0), iterations=jnp.uint32(start_iteration)
            )
        )
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=10, enable_async_checkpointing=False
        )
        with CheckpointManager(tmp_path / "checkpoints", options=options) as manager:
            workflow.checkpoint_manager = manager
            workflow.learn(state)
            assert manager.latest_step() == 3
            restored = manager.restore(3, state)
            assert int(restored.metrics.iterations) == 3


@pytest.mark.parametrize("algorithm", ["ppo", "td3"])
@pytest.mark.parametrize("abstract_template", [False, True])
def test_nnx_model_and_optimizer_checkpoint_preserves_training(
    tmp_path, algorithm, abstract_template
):
    """Restore native variable metadata, targets and Adam moments for resuming."""
    env = ToyEnv(2)
    factory = (
        partial(build_agent, PPOAgent)
        if algorithm == "ppo"
        else partial(build_agent, TD3Agent)
    )
    agent = factory(
        env.obs_space,
        env.action_space,
        actor_hidden_layer_sizes=(4,),
        critic_hidden_layer_sizes=(4,),
    )
    agent_state = agent.init(env.obs_space, env.action_space, jax.random.key(0))
    models = nnx.merge(agent_state.model_state, copy=True)
    policy_name = "policy" if algorithm == "ppo" else "actor"
    tx = optax.adam(1e-3)
    optimizer = nnx.Optimizer(getattr(models, policy_name), tx, wrt=nnx.Param)
    state = State(
        agent_state=agent_state,
        opt_state=nnx.unpack(optimizer),
        key=jax.random.key(1),
    )
    obs = jnp.ones((2, 3))

    @jax.jit
    def update(state):
        models = nnx.merge(state.agent_state.model_state, copy=True)
        policy = getattr(models, policy_name)
        optimizer = nnx.merge(state.opt_state, copy=True)
        _, grads = nnx.value_and_grad(lambda model: jnp.square(model(obs) - 1).mean())(
            policy
        )
        optimizer.update(policy, grads)
        return state.replace(
            agent_state=state.agent_state.replace(model_state=nnx.unpack(models)),
            opt_state=nnx.unpack(optimizer),
        )

    trained = update(state)
    save(tmp_path / "native_state", trained)
    template = (
        jax.tree.map(to_shape_dtype_struct, trained) if abstract_template else trained
    )
    restored = load(tmp_path / "native_state", template)
    options = ocp.CheckpointManagerOptions(enable_async_checkpointing=False)
    with CheckpointManager(tmp_path / "manager", options=options) as manager:
        manager.save(1, trained, force=True)
        managed = manager.restore(1, template)
    chex.assert_trees_all_equal(managed.agent_state, restored.agent_state)
    chex.assert_trees_all_equal(managed.opt_state, restored.opt_state)
    assert isinstance(restored.opt_state.step, nnx.OptState)
    restored_models = nnx.merge(restored.agent_state.model_state, copy=True)
    assert all(
        isinstance(variable, nnx.Param)
        for _, variable in nnx.to_flat_state(
            nnx.state(getattr(restored_models, policy_name))
        )
    )
    chex.assert_trees_all_equal(restored.agent_state, trained.agent_state)
    chex.assert_trees_all_equal(restored.opt_state, trained.opt_state)
    expected = update(trained)
    actual = update(restored)
    chex.assert_trees_all_equal(actual.agent_state, expected.agent_state)
    chex.assert_trees_all_equal(actual.opt_state, expected.opt_state)
    assert int(actual.opt_state.step[...]) == 2
    if algorithm == "td3":
        # Updating the restored actor must leave both target networks untouched.
        chex.assert_trees_all_equal(
            actual.agent_state.model_state.target_actor,
            state.agent_state.model_state.target_actor,
        )
        chex.assert_trees_all_equal(
            actual.agent_state.model_state.target_critic,
            state.agent_state.model_state.target_critic,
        )
