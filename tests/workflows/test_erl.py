"""NNX updates with population actors and shared or independent critics."""

from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

import jax.numpy as jnp
import chex
import jax
from flax import nnx
import numpy as np
import optax
import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf

from evorl.algorithms.td3 import TD3Agent
from evorl.algorithms.erl.erl_td3.erl_td3_workflow import build_erl_rl_update_fn
from evorl.algorithms.erl.cemrl_td3.cemrl_td3_workflow import build_cemrl_rl_update_fn
from evorl.algorithms.contrib.pop_episodic_td3 import build_rl_update_fn
from evorl.agent import AgentState
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.algorithms.erl.erl_td3.erl_td3_workflow import replace_td3_actor_params
from evorl.envs import Box

from tests.helpers.agents import build_agent
from tests.helpers.envs import ToyEnv

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "algorithm",
    [
        "cemrl",
        "cemrl-openes",
        "cemrl-ori",
        "erl-ga",
        "erl-openes",
        "erl-ves",
        "erl-ori",
        "pop-ep-td3",
    ],
)
@pytest.mark.parametrize("norm", ["none", "spectral_norm"])
def test_population_training_preserves_input_and_updates_critic(algorithm, norm, jit):
    root = Path(__file__).resolve().parents[2]
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(
            root
            / f"configs/agent/{'contrib' if algorithm == 'pop-ep-td3' else 'erl'}/{algorithm}.yaml"
        ),
        {
            "env": {"max_episode_steps": 2},
            "num_envs": 1,
            "num_eval_envs": 2,
            "eval_episodes": 2,
            "pop_size": 6,
            "num_elites": 2,
            "num_learning_offspring": 6 if algorithm == "pop-ep-td3" else 2,
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
    )
    base = get_class(config.workflow_cls)
    cls = type("NNXPopulationWorkflow", (base,), {})
    with ExitStack() as patches:
        patches.enter_context(
            patch("jax.local_devices", return_value=jax.local_devices()[:1])
        )
        for module in (
            "evorl.algorithms.erl.cemrl_td3.cemrl",
            "evorl.algorithms.erl.cemrl_td3.cemrl_openes",
            "evorl.algorithms.erl.erl_td3.erl_ga",
            "evorl.algorithms.erl.erl_td3.erl_eda",
            "evorl.algorithms.erl.erl_td3.erl_es",
            "evorl.algorithms.contrib.pop_episodic_td3",
            "evorl.algorithms.erl.cemrl_workflow",
            "evorl.algorithms.erl.erl_workflow",
        ):
            patches.enter_context(
                patch(
                    f"{module}.create_env",
                    side_effect=lambda _, parallel, **kw: ToyEnv(parallel),
                )
            )
        workflow = cls.build_from_config(config, multi_device=False, jit=jit)
        state = workflow.setup(jax.random.key(7))
    old_state = [np.array(x) for x in jax.tree.leaves(state.agent_state)]
    metrics, updated = workflow.step(state)
    expected_iterations = 1
    if algorithm == "cemrl-ori":
        # Its training budget uses the previous iteration's collected steps;
        # initialization has not collected a population episode yet.
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
    replace = jax.jit(replace_td3_actor_params) if jit else replace_td3_actor_params
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
    for before, after in zip(old_leaves, jax.tree.leaves(initial), strict=True):
        np.testing.assert_array_equal(before, after)


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
        model_state=root_axes if shared else 0, obs_preprocessor_state=None
    )
    factory = {
        "erl": build_erl_rl_update_fn,
        "pop-ep": build_rl_update_fn,
        "cemrl": build_cemrl_rl_update_fn,
    }[kind]

    def make_update(size):
        config = OmegaConf.create(
            dict(
                num_rl_agents=size,
                num_learning_offspring=size,
                actor_update_interval=2,
                tau=0.005,
            )
        )
        return factory(agent, tx, config, axes)

    group_update, single_update = make_update(2), make_update(1)
    obs = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
    batch = SampleBatch(
        obs=obs,
        actions=jnp.full((4, 1), 0.2),
        rewards=jnp.ones(4),
        extras=PyTreeDict(
            env_extras=PyTreeDict(ori_obs=obs + 0.1, termination=jnp.zeros(4))
        ),
    )
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, 2, *x.shape)), batch)

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
        model_state=dict(actor=0, target_actor=0, critic=None, target_critic=None),
        obs_preprocessor_state=None,
    )
    update = build_cemrl_rl_update_fn(agent, tx, config, axes)
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
