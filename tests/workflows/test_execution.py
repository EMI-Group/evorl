"""Explicit state layouts and population selection across workflow templates."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evorl.agent import AgentState
from evorl.algorithms.meta.pbt_ppo.pbt_cso_param_ppo import PBTCSOParamPPOWorkflow
from evorl.algorithms.meta.pbt_workflow import (
    PBTOffpolicyWorkflowTemplate,
    PBTOptState,
    PBTWorkflowTemplate,
)
from evorl.algorithms.random_agent import RandomAgentWorkflow
from evorl.distributed import META_POP_AXIS_NAME, shmap_vmap, tree_scatter_set
from evorl.metrics import EvaluateMetric, WorkflowMetric
from evorl.replay_buffers import ReplayBufferState
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict, State
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_get
from evorl.workflows.ec_workflow import ECWorkflowTemplate
from evorl.workflows.rl_workflow import OffPolicyWorkflow, OnPolicyWorkflow
from omegaconf import OmegaConf

from tests.helpers.sharding import compiled_collectives


class _SetupEnv:
    obs_space = None
    action_space = None

    def reset(self, key):
        keys = jax.random.split(key, 2)
        return PyTreeDict(
            key=keys, obs=jax.vmap(lambda k: jax.random.normal(k, (3,)))(keys)
        )


@pytest.fixture
def setup_components(dp_mesh):
    common = {
        "is_multi_device": True,
        "mesh": dp_mesh,
        "dp_axis_name": "DP",
        "env": _SetupEnv(),
        "_setup_agent_and_optimizer": lambda key: (
            AgentState(model_state=PyTreeDict(w=jnp.ones(3))),
            PyTreeDict(m=jnp.zeros(3)),
        ),
        "_setup_workflow_metrics": WorkflowMetric,
    }
    return common


def test_onpolicy_setup_concatenates_local_environments(setup_components):
    self = SimpleNamespace(**setup_components)
    state = jax.jit(lambda key: OnPolicyWorkflow.setup(self, key))(jax.random.key(0))
    assert state.key.shape == ()
    assert state.key.sharding.spec == jax.P()
    assert state.env_state.obs.shape == (4, 3)
    assert state.env_state.key.shape == (4,)
    assert all(
        shard.data.shape == (2, 3) for shard in state.env_state.obs.addressable_shards
    )
    assert state.agent_state.model_state.w.shape == (3,)


def test_offpolicy_setup_keeps_independent_ring_metadata(setup_components):
    class OffPolicyHarness(OffPolicyWorkflow):
        def __init__(self):
            self.__dict__.update(setup_components)

        def _setup_replaybuffer(self, key):
            return ReplayBufferState(data=jnp.zeros((4, 3)))

        def _postsetup_replaybuffer(self, state):
            return state.replace(
                replay_buffer_state=state.replay_buffer_state.replace(
                    buffer_size=state.replay_buffer_state.buffer_size + 1
                ),
                metrics=state.metrics.replace(
                    sampled_timesteps=jax.lax.psum(jnp.uint32(1), "DP")
                ),
            )

    OffPolicyHarness.enable_shmap()
    OffPolicyHarness.enable_jit()
    self = OffPolicyHarness()
    state = self.setup(jax.random.key(1))
    assert state.env_state.obs.shape == (4, 3)
    np.testing.assert_array_equal(state.replay_buffer_state.buffer_size, np.ones(2))
    assert state.replay_buffer_state.data.shape == (8, 3)
    assert state.agent_state.model_state.w.shape == (3,)
    assert state.metrics.sampled_timesteps.shape == ()
    assert int(state.metrics.sampled_timesteps) == 2


def test_ec_setup_assigns_worker_keys_and_ranks(setup_components):
    self = SimpleNamespace(**setup_components, _postsetup=lambda state: state)
    state = ECWorkflowTemplate.setup(self, jax.random.key(2))
    assert state.key.sharding.spec == jax.P("DP")
    np.testing.assert_array_equal(state.distributed_info.rank, [0, 1])


def test_random_setup_replicates_the_root_key(setup_components, dp_mesh):
    self = SimpleNamespace(
        is_multi_device=True,
        mesh=dp_mesh,
        dp_axis_name="DP",
        evaluator=SimpleNamespace(env=setup_components["env"]),
        agent=SimpleNamespace(
            init=lambda *args: AgentState(model_state=PyTreeDict(w=jnp.ones(3)))
        ),
    )
    state = RandomAgentWorkflow.setup(self, jax.random.key(3))
    assert state.key.shape == ()
    assert state.key.sharding.spec == jax.P()


@pytest.mark.integration
@pytest.mark.parametrize(
    "workflow_kind", ["standard", "cso", "shared_buffer", "private_buffer"]
)
def test_population_selection_matches_single_device(workflow_kind, meta_mesh, request):
    class Target:
        def setup(self, key):
            return State(
                key=key,
                agent_state=AgentState(
                    model_state=PyTreeDict(w=jax.random.normal(key, (3,)))
                ),
                metrics=WorkflowMetric(),
            )

        def step(self, state):
            return PyTreeDict(
                loss=state.agent_state.model_state.w.sum()
            ), state.replace(
                agent_state=state.agent_state.replace(
                    model_state=PyTreeDict(w=state.agent_state.model_state.w + 1)
                ),
                metrics=state.metrics.replace(
                    sampled_timesteps=state.metrics.sampled_timesteps + 1
                ),
            )

        def evaluate(self, state):
            return EvaluateMetric(
                episode_return=state.agent_state.model_state.w.sum(),
                episode_length=jnp.float32(1),
            ), state

    class Harness(PBTWorkflowTemplate):
        def _customize_optimizer(self):
            pass

        def _setup_pop_and_pbt_optimizer(self, key):
            return PyTreeDict(
                actor_loss_weight=jnp.arange(4, dtype=jnp.float32) + 1
            ), PBTOptState()

        def apply_hyperparams_to_workflow_state(self, state, hp):
            return state.replace(
                agent_state=state.agent_state.replace(
                    model_state=PyTreeDict(
                        w=state.agent_state.model_state.w + 0 * hp.actor_loss_weight
                    )
                )
            )

    class CSOHarness(PBTCSOParamPPOWorkflow):
        _customize_optimizer = Harness._customize_optimizer
        apply_hyperparams_to_workflow_state = (
            Harness.apply_hyperparams_to_workflow_state
        )

    cfg = OmegaConf.create(
        {
            "checkpoint": {"enable": False},
            "pop_size": 4,
            "workflow_steps_per_iter": 1,
            "warmup_steps": 0,
            "bottom_ratio": 0.25,
            "top_ratio": 0.25,
            "perturb_factor": {"actor_loss_weight": 0.1},
            "search_space": {"actor_loss_weight": {"low": 0.1, "high": 5}},
            "eval_episodes": 1,
            "shared_replay_buffer": workflow_kind == "shared_buffer",
        }
    )

    class Buffer:
        def add(self, state, trajectory):
            return state.replace(
                total=state.total + trajectory.obs.sum(),
                buffer_size=state.buffer_size + trajectory.obs.shape[0],
            )

    class OffTarget(Target):
        replay_buffer = Buffer()

        def setup(self, key):
            return super().setup(key).replace(replay_buffer_state=None)

        def _setup_replaybuffer(self, key):
            return PyTreeDict(total=jnp.float32(0), buffer_size=jnp.int32(0))

        def step(self, state):
            metrics, state = super().step(state)
            return metrics.replace(
                trajectory=SampleBatch(obs=state.agent_state.model_state.w[None])
            ), state

    class OffHarness(PBTOffpolicyWorkflowTemplate):
        _customize_optimizer = Harness._customize_optimizer
        _setup_pop_and_pbt_optimizer = Harness._setup_pop_and_pbt_optimizer
        apply_hyperparams_to_workflow_state = (
            Harness.apply_hyperparams_to_workflow_state
        )

        def _postsetup_replaybuffer(self, state):
            return state

    evaluator = SimpleNamespace(
        evaluate=lambda agent_state, key, num_episodes: EvaluateMetric(
            episode_return=agent_state.model_state.w.sum()[None],
            episode_length=jnp.ones(1),
        )
    )

    mesh = meta_mesh
    cls = {
        "standard": Harness,
        "cso": CSOHarness,
        "shared_buffer": OffHarness,
        "private_buffer": OffHarness,
    }[workflow_kind]
    wf = cls(OffTarget() if cls is OffHarness else Target(), evaluator, cfg)
    wf.mesh = mesh
    request.addfinalizer(wf.close)
    state = jax.jit(wf.setup)(jax.random.key(7))
    for _ in range(2):
        metrics, state = jax.jit(wf.step)(state)
        assert metrics.pop_episode_returns.shape == (4,)
        assert metrics.pop_episode_lengths.shape == (4,)
    metrics, state = jax.jit(wf.evaluate)(state)
    assert state.pop.actor_loss_weight.sharding.spec == jax.P(META_POP_AXIS_NAME)
    assert state.pop_workflow_state.agent_state.model_state.w.sharding.spec == jax.P(
        META_POP_AXIS_NAME, None
    )
    assert metrics.pop_episode_returns.shape == (4,)
    jax.block_until_ready(state)
    if cls is OffHarness:
        if cfg.shared_replay_buffer:
            assert int(state.replay_buffer_state.buffer_size) == 8
            assert state.replay_buffer_state.total.sharding.spec == jax.P()
        else:
            np.testing.assert_array_equal(
                state.replay_buffer_state.buffer_size, [2] * 4
            )
            assert state.replay_buffer_state.total.sharding.spec == jax.P(
                META_POP_AXIS_NAME
            )

    # A one-device run is the numerical reference for distributed selection,
    # mutation, state transfer and shared-buffer accumulation.
    reference = cls(OffTarget() if cls is OffHarness else Target(), evaluator, cfg)
    request.addfinalizer(reference.close)
    reference.mesh = jax.make_mesh(
        (1,),
        (META_POP_AXIS_NAME,),
        axis_types=(jax.sharding.AxisType.Explicit,),
        devices=jax.devices()[:1],
    )
    with jax.set_mesh(reference.mesh):
        expected = jax.jit(reference.setup)(jax.random.key(7))
        for _ in range(2):
            _, expected = jax.jit(reference.step)(expected)
        _, expected = jax.jit(reference.evaluate)(expected)
    assert jax.tree.structure(state) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(state), jax.tree.leaves(expected), strict=True
    ):
        if jax.dtypes.issubdtype(actual_leaf.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(
                jax.random.key_data(actual_leaf), jax.random.key_data(expected_leaf)
            )
        else:
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-5, atol=1e-6)


def test_population_gather_and_owner_scatter(meta_mesh):
    source = jax.device_put(
        jnp.array([10.0, 20.0, 30.0, 40.0]), jax.P(META_POP_AXIS_NAME)
    )
    selected = tree_get(source, jnp.array([3, 0, 3]), out_sharding=jax.P())
    np.testing.assert_array_equal(selected, [40, 10, 40])
    result = jax.jit(
        lambda x: tree_scatter_set(
            x,
            jnp.array([100.0, 400.0]),
            jnp.array([1, 3]),
            jax.P(META_POP_AXIS_NAME),
        )
    )(source)
    np.testing.assert_array_equal(result, [10, 100, 30, 400])


def test_shared_observations_update_each_members_statistics(meta_mesh):
    histories = [np.full((i + 1, 3), i, dtype=np.float32) for i in range(4)]
    members = [
        running_statistics.update(
            running_statistics.init_state(jnp.zeros(3)), jnp.asarray(history)
        )
        for history in histories
    ]
    norm = jax.device_put(
        jax.tree.map(lambda *leaves: jnp.stack(leaves), *members),
        jax.P(META_POP_AXIS_NAME),
    )
    data = jax.device_put(jnp.arange(18, dtype=jnp.float32).reshape(6, 3), jax.P())
    update = jax.jit(
        shmap_vmap(
            running_statistics.update,
            in_specs=(jax.P(META_POP_AXIS_NAME), jax.P()),
            out_specs=jax.P(META_POP_AXIS_NAME),
            vmap_in_axes=(0, None),
        )
    )
    updated = update(norm, data)
    counts, means, stds = map(np.asarray, (updated.count, updated.mean, updated.std))
    for i, history in enumerate(histories):
        observations = np.concatenate((history, np.asarray(data)))
        assert counts[i] == len(observations)
        np.testing.assert_allclose(means[i], observations.mean(0), rtol=1e-6)
        np.testing.assert_allclose(stds[i], observations.std(0), rtol=1e-6)
    for leaf in jax.tree.leaves(updated):
        assert leaf.sharding.spec[0] == META_POP_AXIS_NAME
    assert not compiled_collectives(update.lower(norm, data).compile().as_text())
