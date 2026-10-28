"""NNX updates with population actors and shared or independent critics."""

from pathlib import Path
from contextlib import ExitStack
from functools import partial
from unittest.mock import Mock, create_autospec, patch

import jax.numpy as jnp
import chex
import jax
from flax import nnx
import numpy as np
import optax
import pytest
from hydra.utils import get_class
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from evorl.algorithms.td3 import TD3Agent, TD3TrainMetric
from evorl.algorithms.erl.erl_td3.erl_workflow import update_erl_td3
from evorl.algorithms.erl.cemrl_td3.cemrl_workflow import (
    update_cemrl_td3,
    CEMRLWorkflowBase,
)
from evorl.algorithms.contrib.pop_episodic_td3 import update_pop_td3
from evorl.agent import AgentState
from evorl.networks.utils import model_state_axes
from evorl.metrics import WorkflowMetric, EvaluateMetric
from evorl.utils.orbax_utils import DummyCheckpointManager
from evorl.algorithms.erl.erl_td3.erl_workflow import ERLTrainMetric
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict, State
from evorl.algorithms.erl.cemrl_td3.cemrl_workflow import replace_cemrl_actor_params
from evorl.algorithms.erl.erl_td3.erl_workflow import make_erl_rollout_state
from evorl.algorithms.contrib.pop_episodic_td3 import PopEpisodicTD3Workflow
from evorl.algorithms import erl
from evorl.envs import Box

from tests.helpers.agents import build_agent
from tests.helpers.envs import ToyEnv

pytestmark = pytest.mark.integration


def test_core_workflow_exports_and_independent_td3_population():
    assert set(erl.__all__) == {
        "CEMRLWorkflow",
        "CEMRLOpenESWorkflow",
        "ERLGAWorkflow",
        "ERLESWorkflow",
    }
    assert erl.CEMRLWorkflow.__bases__ == (CEMRLWorkflowBase,)
    assert erl.CEMRLOpenESWorkflow.__bases__ == (CEMRLWorkflowBase,)
    assert "step" in CEMRLWorkflowBase.__abstractmethods__
    assert not issubclass(PopEpisodicTD3Workflow, CEMRLWorkflowBase)


@pytest.mark.parametrize(
    "agent",
    [
        "erl/cemrl",
        "erl/erl-ga",
        "erl/erl-es",
        "erl/cemrl-openes",
        "contrib/cemrl-ori",
        "contrib/erl-openes",
        "contrib/erl-ori",
        "contrib/pop-ep-td3",
        "exp/contrib/cemrl-ori/fastpbrl",
        "exp/contrib/erl-openes/brax-default",
        "exp/contrib/erl-ori/ant",
        "exp/contrib/erl-ori/halfcheetah",
        "exp/contrib/erl-ori/hopper",
        "exp/contrib/erl-ori/walker2d",
    ],
)
def test_reorganized_configuration_resolves_workflow(agent):
    root = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        config = compose(config_name="config", overrides=[f"agent={agent}"])
    cls = get_class(config.workflow_cls)
    if agent.startswith("erl/"):
        assert cls in (
            erl.CEMRLWorkflow,
            erl.CEMRLOpenESWorkflow,
            erl.ERLGAWorkflow,
            erl.ERLESWorkflow,
        )
    else:
        assert cls.__module__.startswith("evorl.algorithms.contrib.")
    if agent == "contrib/pop-ep-td3":
        assert "num_learning_offspring" not in config


@pytest.fixture
def population_workflow():
    workflows = []

    def build(algorithm, norm="none", jit=True, **overrides):
        root = Path(__file__).resolve().parents[2]
        config = OmegaConf.merge(
            OmegaConf.load(root / "configs/config.yaml"),
            OmegaConf.load(
                root
                / f"configs/agent/{'contrib' if algorithm in ('cemrl-ori', 'erl-openes', 'erl-ori', 'pop-ep-td3') else 'erl'}/{algorithm}.yaml"
            ),
            {
                "env": {"max_episode_steps": 2},
                "num_envs": 1,
                "num_eval_envs": 2,
                "eval_episodes": 2,
                "pop_size": 6,
                "num_elites": 2,
                **(
                    dict(num_learning_offspring=2)
                    if algorithm.startswith("cemrl")
                    else {}
                ),
                "num_rl_agents": 2,
                "warmup_iters": 0,
                "random_timesteps": 4,
                "batch_size": 2,
                "replay_buffer_capacity": 32,
                "num_rl_updates_per_iter": 2,
                "actor_update_interval": 2,
                "normalize_obs": True,
                "agent_network": {
                    "norm_layer_type": norm,
                    "critic_hidden_layer_sizes": [4],
                    "actor_hidden_layer_sizes": [4],
                },
            },
            overrides,
        )
        base = get_class(config.workflow_cls)
        cls = type("NNXPopulationWorkflow", (base,), {})
        with ExitStack() as patches:
            patches.enter_context(
                patch("jax.local_devices", return_value=jax.local_devices()[:1])
            )
            for module in (
                "evorl.algorithms.erl.cemrl_td3.cemrl_workflow",
                "evorl.algorithms.contrib.pop_episodic_td3",
                "evorl.algorithms.erl.erl_td3.erl_workflow",
            ):
                patches.enter_context(
                    patch(
                        f"{module}.create_env",
                        side_effect=lambda _, parallel, **kw: ToyEnv(parallel),
                    )
                )
            workflow = cls.build_from_config(config, multi_device=False, jit=jit)
            state = workflow.setup(jax.random.key(7))
        workflows.append(workflow)
        return workflow, state

    yield build
    for workflow in workflows:
        workflow.close()


@pytest.mark.parametrize(
    "algorithm",
    [
        "cemrl",
        "cemrl-openes",
        "cemrl-ori",
        "erl-ga",
        "erl-openes",
        "erl-es",
        "erl-ori",
        "pop-ep-td3",
    ],
)
@pytest.mark.parametrize("norm", ["none", "spectral_norm"])
def test_population_training_preserves_input_and_updates_critic(
    algorithm, norm, jit, population_workflow
):
    workflow, state = population_workflow(algorithm, norm, jit)
    config = workflow.config
    assert not hasattr(workflow, "_rl_update_fn")
    assert not hasattr(workflow, "_rl_sample_and_update_fn")
    if algorithm == "pop-ep-td3":
        assert not hasattr(workflow, "ec_optimizer")
        assert "ec_opt_state" not in state
        assert "num_learning_offspring" not in workflow.config
    old_state = [np.array(x) for x in jax.tree.leaves(state.agent_state)]
    metrics, updated = workflow.step(state)
    evaluation, _ = workflow.evaluate(updated)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(evaluation))
    expected_iterations = 1
    if algorithm == "cemrl-ori":
        # Its training budget uses the previous iteration's collected steps;
        # initialization has not collected a population episode yet.
        assert float(metrics.rl_metrics.actor_loss) == 0
        assert float(metrics.rl_metrics.critic_loss) == 0
        assert int(nnx.merge(updated.opt_state.critic, copy=True).step[...]) == 0
        metrics, updated = workflow.step(updated)
        expected_iterations = 2
    jax.block_until_ready(updated)
    for before, after in zip(
        old_state, jax.tree.leaves(state.agent_state), strict=True
    ):
        np.testing.assert_array_equal(before, after)
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(
                nnx.state(
                    nnx.merge(state.agent_state.model_state, copy=True).critic,
                    nnx.Param,
                )
            ),
            jax.tree.leaves(
                nnx.state(
                    nnx.merge(updated.agent_state.model_state, copy=True).critic,
                    nnx.Param,
                )
            ),
        )
    )
    assert int(updated.metrics.iterations) == expected_iterations
    episodes_per_iter = config.pop_size * config.episodes_for_fitness
    timesteps_per_iter = metrics.pop_episode_length.sum() * config.episodes_for_fitness
    if isinstance(metrics, ERLTrainMetric) and metrics.rl_episode_length is not None:
        num_learners = 1 if algorithm == "erl-openes" else config.num_rl_agents
        episodes_per_iter += num_learners * config.rollout_episodes
        timesteps_per_iter += metrics.rl_episode_length.sum() * config.rollout_episodes
    assert (
        int(updated.metrics.sampled_episodes) == expected_iterations * episodes_per_iter
    )
    assert int(updated.metrics.sampled_timesteps) == expected_iterations * int(
        timesteps_per_iter
    )
    for loss_name in ("actor_loss", "critic_loss"):
        np.testing.assert_allclose(
            getattr(metrics.rl_metrics, loss_name),
            metrics.rl_metrics.raw_loss_dict[loss_name].mean(),
            rtol=1e-5,
            atol=1e-6,
        )
    for leaf in jax.tree.leaves(metrics):
        assert np.all(np.isfinite(leaf))
    if algorithm.startswith("cemrl"):
        assert not jax.tree.leaves(
            nnx.state(
                nnx.merge(updated.agent_state.model_state, copy=True).actor, nnx.Param
            )
        )
        assert updated.opt_state.actor is None
    elif algorithm != "erl-openes":
        actor = jax.tree.leaves(
            nnx.state(
                nnx.merge(updated.agent_state.model_state, copy=True).actor, nnx.Param
            )
        )[0]
        assert actor.shape[0] == (
            config.pop_size if algorithm == "pop-ep-td3" else config.num_rl_agents
        )

    if norm == "spectral_norm":
        for params in (
            (state.ec_opt_state.mean,)
            if algorithm.startswith("cemrl")
            else (
                nnx.state(
                    nnx.merge(updated.agent_state.model_state, copy=True).actor,
                    nnx.Param,
                ),
            )
        ):
            assert all(
                isinstance(value, nnx.Param) for _, value in nnx.to_flat_state(params)
            )
        assert jax.tree.leaves(
            nnx.state(
                nnx.merge(updated.agent_state.model_state, copy=True).critic,
                nnx.Not(nnx.Param),
            )
        )
        assert any(
            not np.array_equal(before, after)
            for before, after in zip(
                jax.tree.leaves(
                    nnx.state(
                        nnx.merge(state.agent_state.model_state, copy=True).critic,
                        nnx.Not(nnx.Param),
                    )
                ),
                jax.tree.leaves(
                    nnx.state(
                        nnx.merge(updated.agent_state.model_state, copy=True).critic,
                        nnx.Not(nnx.Param),
                    )
                ),
                strict=True,
            )
        )


@pytest.mark.parametrize("algorithm", ["erl-ga", "erl-es", "erl-openes"])
def test_erl_setup_warmup_collects_episodes_without_updating_rl_learners(
    algorithm, jit, population_workflow
):
    with patch("evorl.recorders.ChainRecorder.write") as write:
        workflow, state = population_workflow(
            algorithm,
            norm="spectral_norm",
            jit=jit,
            warmup_iters=3,
            eval_interval=2,
            random_timesteps=0,
        )
    write.assert_not_called()
    agent_key = jax.random.split(jax.random.key(7), 3)[1]
    initial_agent, initial_opt, _ = workflow._setup_agent_and_optimizer(agent_key)
    chex.assert_trees_all_equal(state.agent_state, initial_agent)
    chex.assert_trees_all_equal(state.opt_state, initial_opt)
    assert int(state.metrics.iterations) == 3
    assert int(state.metrics.sampled_episodes) == 3 * workflow.config.pop_size
    assert int(state.metrics.sampled_timesteps) == 6 * workflow.config.pop_size
    assert int(state.metrics.rl_sampled_timesteps) == 0
    assert (
        int(state.replay_buffer_state.buffer_size)
        == workflow.config.replay_buffer_capacity
    )
    # A normal training iteration can consume the replay produced by EC warmup.
    metrics, trained = workflow.step(state)
    assert int(trained.metrics.iterations) == 4
    assert int(trained.metrics.rl_sampled_timesteps) > 0
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(metrics))


def test_imported_td3_population_resizes_auxiliary_state_without_evolving_it(jit):

    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = Box(low=-jnp.ones(2), high=jnp.ones(2))
    agent = build_agent(
        TD3Agent,
        obs_space,
        action_space,
        actor_hidden_layer_sizes=(4,),
        critic_hidden_layer_sizes=(4,),
        norm_layer_type="spectral_norm",
    )
    initial = agent.init(obs_space, action_space, jax.random.key(0))
    old_leaves = [np.array(x) for x in jax.tree.leaves(initial)]
    models = nnx.merge(initial.model_state, copy=True)
    params = nnx.state(models.actor, nnx.Param)
    auxiliary = nnx.state(models.actor, nnx.Not(nnx.Param))
    critic = nnx.state(models.critic)
    target_critic = nnx.state(models.target_critic)
    replace = jax.jit(replace_cemrl_actor_params) if jit else replace_cemrl_actor_params
    genomes = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, 3, *x.shape)), params)
    imported = replace(initial, genomes)
    imported_models = nnx.merge(imported.model_state, copy=True)
    chex.assert_trees_all_equal(nnx.state(imported_models.actor, nnx.Param), genomes)
    chex.assert_trees_all_equal(
        nnx.state(imported_models.target_actor, nnx.Param), genomes
    )
    chex.assert_trees_all_equal(
        nnx.state(imported_models.actor, nnx.Not(nnx.Param)),
        jax.tree.map(lambda x: jnp.broadcast_to(x, (2, 3, *x.shape)), auxiliary),
    )
    changed_auxiliary = jax.tree.map(
        lambda x: x + 0.2, nnx.state(imported_models.actor, nnx.Not(nnx.Param))
    )
    nnx.update(imported_models.actor, changed_auxiliary)
    changed = imported.replace(model_state=nnx.unpack(imported_models))
    resized = replace(
        changed, jax.tree.map(lambda x: jnp.broadcast_to(x, (4, *x.shape)), params)
    )
    dropped = replace(resized, None)
    dropped_models = nnx.merge(dropped.model_state, copy=True)
    assert not jax.tree.leaves(nnx.state(dropped_models.actor, nnx.Param))
    chex.assert_trees_all_close(
        nnx.state(dropped_models.actor, nnx.Not(nnx.Param)),
        jax.tree.map(lambda x: x + 0.2, auxiliary),
    )
    restored = replace(
        replace(
            dropped, jax.tree.map(lambda x: jnp.broadcast_to(x, (5, *x.shape)), params)
        ),
        params,
    )
    restored_models = nnx.merge(restored.model_state, copy=True)
    chex.assert_trees_all_close(
        nnx.state(restored_models.actor, nnx.Not(nnx.Param)),
        jax.tree.map(lambda x: x + 0.2, auxiliary),
    )
    chex.assert_trees_all_equal(nnx.state(restored_models.critic), critic)
    chex.assert_trees_all_equal(nnx.state(restored_models.target_critic), target_critic)
    chex.assert_trees_all_equal(
        nnx.state(restored_models.target_actor), nnx.state(restored_models.actor)
    )
    for before, after in zip(old_leaves, jax.tree.leaves(initial), strict=True):
        np.testing.assert_array_equal(before, after)


def test_erl_evolutionary_rollouts_use_actor_only_without_mutating_learners(jit):
    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = Box(low=-jnp.ones(1), high=jnp.ones(1))
    agent = build_agent(
        TD3Agent,
        obs_space,
        action_space,
        actor_hidden_layer_sizes=(4,),
        critic_hidden_layer_sizes=(4,),
        norm_layer_type="spectral_norm",
        normalize_obs=True,
    )
    state = agent.init(obs_space, action_space, jax.random.key(31))
    before = jax.tree.map(np.array, state)
    models = agent.get_models(state, copy=True)
    params = nnx.state(models.actor, nnx.Param)
    population = jax.tree.map(
        lambda x: jnp.broadcast_to(x + 0.1, (4, *x.shape)), params
    )
    build = jax.jit(make_erl_rollout_state) if jit else make_erl_rollout_state
    rollout_state = build(state, population)
    rollout_models = agent.get_models(rollout_state, copy=True)
    assert set(rollout_models.keys()) == {"actor"}
    chex.assert_trees_all_equal(nnx.state(rollout_models.actor, nnx.Param), population)
    actions, _ = jax.jit(
        jax.vmap(
            agent.evaluate_actions,
            in_axes=(AgentState(model_state=0, obs_preprocessor_state=None), None, 0),
        )
    )(
        rollout_state,
        SampleBatch(obs=jnp.ones((2, 3))),
        jax.random.split(jax.random.key(32), 4),
    )
    assert actions.shape == (4, 2, 1)
    assert np.isfinite(actions).all()
    for old, current in zip(
        jax.tree.leaves(before), jax.tree.leaves(state), strict=True
    ):
        np.testing.assert_array_equal(old, current)


def test_cemrl_population_axes_match_graphstate(jit, population_workflow):
    """NNX filters and their GraphState axis tree collect identical transitions."""
    workflow, state = population_workflow("cemrl", jit=jit)
    axes = workflow.agent_state_vmap_axes
    assert isinstance(axes.model_state, nnx.StateAxes)
    params = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (workflow.config.pop_size, *x.shape)),
        state.ec_opt_state.mean,
    )
    agent_state = replace_cemrl_actor_params(state.agent_state, params)
    axis_tree = model_state_axes(agent_state.model_state, axes.model_state)
    results = []
    for model_axes in (axes.model_state, axis_tree):
        workflow.agent_state_vmap_axes = axes.replace(model_state=model_axes)

        def collect(agent_state, buffer_state, key):
            return workflow._rollout(agent_state, buffer_state, key)

        collect_fn = jax.jit(collect) if jit else collect
        results.append(
            collect_fn(agent_state, state.replay_buffer_state, jax.random.key(13))
        )
    # Uninitialized replay slots are intentionally unspecified.
    for metrics, trajectory, buffer_state in results:
        assert int(buffer_state.buffer_size) > 0
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(metrics))
    chex.assert_trees_all_close(results[0][:2], results[1][:2])


@pytest.mark.parametrize("kind", ["erl", "pop-ep", "cemrl"])
@pytest.mark.parametrize("nested", [False, True])
def test_population_clipping_matches_independent_adam_updates(kind, nested, jit):
    """Each actor/independent critic applies clipping and Adam to its own state.

    For CEMRL, freeze the shared critic's weights to isolate actor updates;
    its power-iteration state still advances through the ordinary loss path.
    """
    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = Box(low=-jnp.ones(1), high=jnp.ones(1))
    agent = build_agent(
        TD3Agent,
        obs_space,
        action_space,
        actor_hidden_layer_sizes=(4,),
        critic_hidden_layer_sizes=(4,),
        norm_layer_type="spectral_norm",
        policy_noise=0.0,
        normalize_obs=False,
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    shared = kind == "cemrl"
    root_axes = dict(
        actor=0,
        target_actor=0,
        critic=None if shared else 0,
        target_critic=None if shared else 0,
    )
    axes = AgentState(
        model_state=nnx.StateAxes(
            (nnx.PathContains(name), axis) for name, axis in root_axes.items()
        )
        if shared
        else 0,
        obs_preprocessor_state=None,
    )
    factory = {
        "erl": update_erl_td3,
        "pop-ep": update_pop_td3,
        "cemrl": update_cemrl_td3,
    }[kind]

    def make_update(size, axes):
        config = OmegaConf.create(
            dict(
                num_rl_agents=size,
                num_learning_offspring=size,
                pop_size=size,
                actor_update_interval=3,
                tau=0.005,
            )
        )
        return partial(factory, agent=agent, agent_state_vmap_axes=axes, config=config)

    obs = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
    batch = SampleBatch(
        obs=obs,
        actions=jnp.full((4, 1), 0.2),
        rewards=jnp.ones(4),
        extras=PyTreeDict(
            env_extras=PyTreeDict(ori_obs=obs + 0.1, termination=jnp.zeros(4))
        ),
    )
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (3, 2, *x.shape)), batch)

    def member(state, index):
        models = nnx.merge(state.model_state, copy=True)
        for name, axis in root_axes.items():
            if axis == 0:
                nnx.update(
                    models[name],
                    jax.tree.map(
                        lambda x: x[index : index + 1], nnx.state(models[name])
                    ),
                )
        return state.replace(model_state=nnx.unpack(models))

    def init_opt(state):
        models = nnx.merge(state.model_state, copy=True)

        def create(model):
            return nnx.Optimizer(model, tx, wrt=nnx.Param)

        return PyTreeDict(
            actor=nnx.unpack(nnx.vmap(create)(models.actor)),
            critic=nnx.unpack(
                nnx.Optimizer(models.critic, optax.set_to_zero(), wrt=nnx.Param)
            )
            if shared
            else nnx.unpack(nnx.vmap(create)(models.critic)),
        )

    def run(key):
        initial = jax.vmap(lambda k: agent.init(obs_space, action_space, k))(
            jax.random.split(key, 2)
        )
        if shared:
            models = nnx.merge(initial.model_state, copy=True)
            for name in ("critic", "target_critic"):
                nnx.update(
                    models[name], jax.tree.map(lambda x: x[0], nnx.state(models[name]))
                )
            initial = initial.replace(model_state=nnx.unpack(models))
        group_update = make_update(2, axes)
        single_update = make_update(1, axes)
        group, group_opt = initial, init_opt(initial)
        singles = [member(initial, i) for i in range(2)]
        single_opts = [init_opt(s) for s in singles]
        for iteration, factors in enumerate(((1.0, 100.0), (100.0, 1.0), (1.0, 100.0))):
            current = batch.replace(
                rewards=batch.rewards * jnp.array(factors)[None, :, None]
            )
            update_key = jax.random.fold_in(key, iteration)
            (group, group_opt), _ = group_update(group, group_opt, current, update_key)
            for i in range(2):
                local = jax.tree.map(
                    lambda x, index=i: x[:, index : index + 1], current
                )
                (singles[i], single_opts[i]), _ = single_update(
                    singles[i], single_opts[i], local, update_key
                )
        return group, group_opt, tuple(singles), tuple(single_opts)

    fn = jax.vmap(run) if nested else run
    fn = jax.jit(fn) if jit else fn
    key = jax.random.key(8)
    result = fn(jax.random.split(key, 2) if nested else key)
    groups = (
        [jax.tree.map(lambda x, index=i: x[index], result) for i in range(2)]
        if nested
        else [result]
    )
    for group, group_opt, singles, single_opts in groups:
        np.testing.assert_array_equal(
            nnx.merge(group_opt.actor, copy=True).step[...], [3, 3]
        )
        np.testing.assert_array_equal(
            nnx.merge(group_opt.critic, copy=True).step[...], 9 if shared else [9, 9]
        )
        for i, (expected, expected_opt) in enumerate(
            zip(singles, single_opts, strict=True)
        ):
            actual = member(group, i)
            actual_opt = group_opt.replace(
                actor=jax.tree.map(
                    lambda x, index=i: x[index : index + 1], group_opt.actor
                ),
                critic=group_opt.critic
                if shared
                else jax.tree.map(
                    lambda x, index=i: x[index : index + 1], group_opt.critic
                ),
            )
            for got, want in zip(
                jax.tree.leaves((actual, actual_opt)),
                jax.tree.leaves((expected, expected_opt)),
                strict=True,
            ):
                np.testing.assert_allclose(got, want, rtol=3e-5, atol=2e-6)


def test_cemrl_shared_critic_clips_the_mean_gradient(jit):
    """The shared critic keeps one optimizer for the mean across actor batches."""
    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = Box(low=-jnp.ones(1), high=jnp.ones(1))
    agent = build_agent(
        TD3Agent,
        obs_space,
        action_space,
        actor_hidden_layer_sizes=(4,),
        critic_hidden_layer_sizes=(4,),
        policy_noise=0.0,
        normalize_obs=False,
    )
    key = jax.random.key(12)
    initial = jax.vmap(lambda k: agent.init(obs_space, action_space, k))(
        jax.random.split(key, 2)
    )
    models = nnx.merge(initial.model_state, copy=True)
    for name in ("critic", "target_critic"):
        nnx.update(models[name], jax.tree.map(lambda x: x[0], nnx.state(models[name])))
    initial = initial.replace(model_state=nnx.unpack(models))
    tx = optax.chain(optax.clip_by_global_norm(0.1), optax.adam(1e-3))
    opt_state = PyTreeDict(
        actor=nnx.unpack(
            nnx.vmap(lambda m: nnx.Optimizer(m, tx, wrt=nnx.Param))(models.actor)
        ),
        critic=nnx.unpack(nnx.Optimizer(models.critic, tx, wrt=nnx.Param)),
    )
    obs = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) / 10
    batch = SampleBatch(
        obs=obs,
        actions=jnp.full((2, 4, 1), 0.2),
        rewards=jnp.array([[1.0] * 4, [100.0] * 4]),
        extras=PyTreeDict(
            env_extras=PyTreeDict(ori_obs=obs + 0.1, termination=jnp.zeros((2, 4)))
        ),
    )
    params = nnx.state(models.critic, nnx.Param)
    _, critic_key, _ = jax.random.split(key, 3)
    keys = jax.random.split(critic_key, 2)

    def mean_loss(critic_params):
        losses = []
        for i in range(2):
            member = nnx.merge(initial.model_state, copy=True)
            for name in ("actor", "target_actor"):
                nnx.update(
                    member[name],
                    jax.tree.map(lambda x, index=i: x[index], nnx.state(member[name])),
                )
            nnx.update(member.critic, critic_params)
            local_batch = jax.tree.map(lambda x, index=i: x[index], batch)
            losses.append(
                agent.critic_loss(
                    initial.replace(model_state=nnx.unpack(member)),
                    local_batch,
                    keys[i],
                ).critic_loss
            )
        return jnp.mean(jnp.stack(losses))

    expected_loss, grads = jax.value_and_grad(mean_loss)(params)
    updates, _ = tx.update(grads, tx.init(params), params)
    expected_params = optax.apply_updates(params, updates)
    config = OmegaConf.create(
        dict(num_learning_offspring=2, actor_update_interval=1, tau=0.005)
    )
    axes = AgentState(
        model_state=nnx.StateAxes(
            {
                nnx.PathContains("actor"): 0,
                nnx.PathContains("target_actor"): 0,
                ...: None,
            }
        ),
        obs_preprocessor_state=None,
    )
    update = partial(
        update_cemrl_td3, agent=agent, agent_state_vmap_axes=axes, config=config
    )
    if jit:
        update = jax.jit(update)
    (updated, next_opt), metrics = update(
        initial, opt_state, jax.tree.map(lambda x: x[None], batch), key
    )
    actual_models = nnx.merge(updated.model_state, copy=True)
    chex.assert_trees_all_close(
        nnx.state(actual_models.critic, nnx.Param),
        expected_params,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(metrics[0], expected_loss, rtol=1e-6)
    assert nnx.merge(next_opt.critic, copy=True).step[...].shape == ()
    np.testing.assert_array_equal(
        nnx.merge(next_opt.actor, copy=True).step[...], [1, 1]
    )


def test_erl_eda_evaluates_candidates_matching_openes_noise(
    population_workflow, monkeypatch
):
    """Fitness indices must describe the same candidates retained by OpenES."""
    workflow, state = population_workflow("erl-openes", jit=False)
    expected_pop, _ = workflow.ec_optimizer.ask(state.ec_opt_state)
    rollout = workflow._ec_rollout
    evaluated = []

    def collect(agent_state, buffer_state, key):
        evaluated.append(
            nnx.state(workflow.agent.get_models(agent_state).actor, nnx.Param)
        )
        return rollout(agent_state, buffer_state, key)

    monkeypatch.setattr(workflow, "_ec_rollout", collect)
    workflow.step(state)
    assert workflow.config.mirror_sampling
    assert len(evaluated) == 1
    chex.assert_trees_all_close(evaluated[0], expected_pop)


@pytest.mark.parametrize(
    "workflow_cls",
    [
        "evorl.algorithms.erl.cemrl_td3.cemrl.CEMRLWorkflow",
        "evorl.algorithms.erl.cemrl_td3.cemrl_openes.CEMRLOpenESWorkflow",
        "evorl.algorithms.erl.erl_td3.erl_ga.ERLGAWorkflow",
        "evorl.algorithms.erl.erl_td3.erl_es.ERLESWorkflow",
        "evorl.algorithms.contrib.cemrl_origin.CEMRLOriginalWorkflow",
        "evorl.algorithms.contrib.erl_origin.ERLOriginalWorkflow",
        "evorl.algorithms.contrib.erl_eda.ERLEDAWorkflow",
        "evorl.algorithms.contrib.pop_episodic_td3.PopEpisodicTD3Workflow",
    ],
)
@pytest.mark.parametrize("save_replay_buffer", [False, True])
def test_erl_learn_records_evaluates_and_saves_final_state(
    workflow_cls, save_replay_buffer
):
    """Exercise each learn loop's logging and checkpoint API without training."""
    cls = get_class(workflow_cls)
    workflow = cls.__new__(cls)
    workflow.config = OmegaConf.create(
        dict(
            total_episodes=1,
            episodes_for_fitness=1,
            rollout_episodes=1,
            pop_size=2,
            num_rl_agents=2,
            eval_interval=10,
            save_replay_buffer=save_replay_buffer,
        )
    )
    workflow.recorder = Mock()
    workflow.checkpoint_manager = create_autospec(DummyCheckpointManager, instance=True)
    initial = State(
        metrics=WorkflowMetric(),
        replay_buffer_state=PyTreeDict(data=jnp.ones(2)),
        ec_opt_state=PyTreeDict(variance=jnp.ones(2)),
    )
    final = initial.replace(
        metrics=initial.metrics.replace(
            iterations=jnp.uint32(1), sampled_episodes=jnp.uint32(4)
        )
    )
    train_metrics = ERLTrainMetric(
        pop_episode_return=jnp.ones(2),
        pop_episode_length=jnp.ones(2),
        rl_episode_return=jnp.ones(2),
        rl_episode_length=jnp.ones(2),
        rl_metrics=TD3TrainMetric(
            actor_loss=jnp.zeros(()),
            critic_loss=jnp.zeros(()),
            raw_loss_dict=PyTreeDict(actor_loss=jnp.zeros(2), critic_loss=jnp.zeros(2)),
        ),
    )
    workflow.step = Mock(side_effect=[(train_metrics, final)])
    workflow.evaluate = Mock(
        return_value=(
            EvaluateMetric(episode_return=jnp.ones(2), episode_length=jnp.ones(2)),
            final,
        )
    )
    assert workflow.learn(initial) is final
    workflow.step.assert_called_once_with(initial)
    workflow.evaluate.assert_called_once_with(final)
    workflow.recorder.write.assert_called()
    workflow.checkpoint_manager.save.assert_called_once()
    args, kwargs = workflow.checkpoint_manager.save.call_args
    assert len(args) == 2
    assert int(args[0]) == 1
    assert kwargs == {"force": True}
    if save_replay_buffer:
        assert args[1] is final
    else:
        assert args[1].replay_buffer_state is None
        assert final.replay_buffer_state is not None
    workflow.step.reset_mock()
    workflow.checkpoint_manager.save.reset_mock()
    assert workflow.learn(final) is final
    workflow.step.assert_not_called()
    workflow.checkpoint_manager.save.assert_not_called()
