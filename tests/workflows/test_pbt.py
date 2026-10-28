"""Exercise real PBT targets on the independently named meta-population axis."""

from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax
from flax import nnx
from evorl.algorithms.meta.pbt_sac.param_sac import ParamSACWorkflow, ParamSACAgent
from evorl.algorithms.meta.pbt_sac.pbt_param_sac import PBTParamSACWorkflow
from evorl.algorithms.meta.pbt import PBTWorkflow
from evorl.types import State
from evorl.algorithms.meta.pbt_workflow import PBTOffpolicyWorkflowTemplate
from evorl.algorithms.offpolicy_utils import OffPolicyWorkflowTemplate
from evorl.distributed import META_POP_AXIS_NAME, POP_AXIS_NAME
from evorl.replay_buffers import ReplayBuffer
from evorl.rollout import eval_rollout_episode, fast_eval_rollout_episode, rollout
from evorl.types import PyTreeDict
from evorl.utils.orbax_utils import save, load
from hydra.utils import get_class
from omegaconf import OmegaConf

from tests.helpers.agents import build_agent
from tests.helpers.envs import ToyEnv
from tests.helpers.sharding import compiled_collectives

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("random_timesteps", [0, 1])
def test_pbt_sac_rejects_warmup_without_a_complete_environment_batch(random_timesteps):
    target = object.__new__(ParamSACWorkflow)
    target.env = ToyEnv(2)
    workflow = object.__new__(PBTOffpolicyWorkflowTemplate)
    workflow.workflow = target
    workflow.config = OmegaConf.create(
        dict(random_timesteps=random_timesteps, target_workflow=dict(num_envs=2))
    )
    with pytest.raises(ValueError, match="random_timesteps must be at least"):
        workflow._postsetup_replaybuffer(
            State(pop_workflow_state=None, replay_buffer_state=None)
        )


def test_shared_replay_buffer_is_replicated_and_accepts_writes(meta_mesh, jit):
    target = object.__new__(OffPolicyWorkflowTemplate)
    target.env = ToyEnv(2)
    target.replay_buffer = ReplayBuffer(capacity=16, sample_batch_size=2)
    workflow = object.__new__(PBTOffpolicyWorkflowTemplate)
    workflow.workflow = target
    workflow.config = OmegaConf.create(dict(shared_replay_buffer=True))
    key = jax.device_put(jax.random.key(3), jax.P())
    initialize = workflow._setup_replaybuffer
    if jit:
        initialize = jax.jit(initialize)
    state = initialize(key)
    assert int(state.current_index) == 0
    assert int(state.buffer_size) == 0
    assert state.data.obs.shape == (16, 3)
    for leaf in jax.tree.leaves(state):
        assert leaf.sharding.is_fully_replicated
        assert len(leaf.addressable_shards) == 2

    samples = jax.tree.map(
        lambda x: jnp.ones((2, *x.shape[1:]), dtype=x.dtype), state.data
    )
    updated = target.replay_buffer.add(state, samples)
    assert int(updated.buffer_size) == 2
    assert int(updated.current_index) == 2
    for leaf in jax.tree.leaves(updated.data):
        for shard in leaf.addressable_shards:
            written = np.asarray(shard.data)[:2]
            np.testing.assert_array_equal(written, np.ones_like(written))
    compiled = jax.jit(workflow._setup_replaybuffer).lower(key).compile()
    assert not compiled_collectives(compiled.as_text())


@pytest.mark.parametrize("rollout_kind", ["train", "episode", "fast"])
def test_rollout_uses_local_population_axes(rollout_kind, meta_mesh):
    class VariableLengthEnv(ToyEnv):
        def step(self, state, action):
            next_state = super().step(state, action)
            # Input-dependent termination keeps the evaluation loop dynamic;
            # uniform lengths can hide automatic cross-device loop reductions.
            threshold = 1 + jnp.floor(state.obs[:, 0] * 4).astype(jnp.int32)
            return next_state.replace(
                done=(next_state.env_state >= threshold).astype(jnp.float32)
            )

    env = VariableLengthEnv(2)

    def action_fn(agent_state, batch, key):
        return batch.obs[:, :1], PyTreeDict()

    def instance(key):
        state = env.reset(key)
        if rollout_kind == "train":
            result, _ = rollout(env.step, action_fn, state, None, key, 2)
            return result.rewards.sum()
        if rollout_kind == "episode":
            result, _ = eval_rollout_episode(env.step, action_fn, state, None, key, 2)
            return result.rewards.sum()
        result, _ = fast_eval_rollout_episode(env.step, action_fn, state, None, key, 2)
        return result.episode_return.sum()

    keys = jax.random.split(jax.random.key(3), 4)
    expected = jax.jit(instance)(keys[0])
    mesh = meta_mesh
    with jax.set_mesh(mesh):
        mapped = jax.shard_map(
            jax.vmap(instance),
            in_specs=jax.P(META_POP_AXIS_NAME),
            out_specs=jax.P(META_POP_AXIS_NAME),
        )
        inputs = jax.device_put(keys, jax.P(META_POP_AXIS_NAME))
        executable = jax.jit(mapped).lower(inputs).compile()
        hlo = executable.as_text()
        assert not compiled_collectives(hlo)
        actual = executable(inputs)
        np.testing.assert_allclose(np.asarray(actual)[0], expected, rtol=1e-6)


@pytest.mark.parametrize("target", ["ppo", "paramppo", "paramopenes"])
def test_pbt_members_train_independently(target):
    root = Path(__file__).resolve().parents[2]
    common = OmegaConf.load(root / "configs/config.yaml")
    target_config = OmegaConf.merge(
        common,
        OmegaConf.load(
            root
            / (
                "configs/agent/ppo.yaml"
                if target == "ppo"
                else f"configs/agent/meta/{target}.yaml"
            )
        ),
        {
            "num_envs": 2,
            "num_eval_envs": 2,
            "eval_episodes": 2,
            "normalize_obs": True,
            "random_timesteps": 4,
            "agent_network": {"actor_hidden_layer_sizes": [4]},
        },
    )
    if target in ("ppo", "paramppo"):
        target_config.rollout_length = 2
        target_config.minibatch_size = 4
        target_config.reuse_rollout_epochs = 1
        target_config.agent_network.critic_hidden_layer_sizes = [4]
    else:
        target_config.num_envs = 1
        target_config.pop_size = 4
        target_config.episodes_for_fitness = 1
    config = OmegaConf.merge(
        common,
        OmegaConf.load(
            root
            / f"configs/agent/meta/pbt-{'paramppo' if target == 'ppo' else target}.yaml"
        ),
        {
            "target_workflow": target_config,
            "pop_size": 4,
            "num_eval_envs": 2,
            "eval_episodes": 2,
            "workflow_steps_per_iter": 1,
            "warmup_steps": 100,
            "bottom_ratio": 0.25,
            "top_ratio": 0.25,
            "env": {"max_episode_steps": 2},
        },
    )
    if target == "ppo":
        config.workflow_cls = "evorl.algorithms.meta.pbt.PBTWorkflow"
        config.search_space = {"lr": {"low": 1e-4, "high": 1e-2}}
        config.perturb_factor = {"lr": 0.2}
    target_cls = get_class(target_config.workflow_cls)
    original_target_step = target_cls.step
    # Both the PBT class and its dynamically loaded member class must be fresh:
    # build_from_config applies JIT transformations to class methods.
    member_cls = type(f"Member{target_cls.__name__}", (target_cls,), {})
    base = get_class(config.workflow_cls)
    cls = type(f"AxisProbe{base.__name__}", (base,), {})

    def create_env(*args, **kwargs):
        return ToyEnv(kwargs["parallel"])

    with (
        patch("hydra.utils.get_class", return_value=member_cls),
        patch(f"{target_cls.__module__}.create_env", side_effect=create_env),
        patch("evorl.algorithms.meta.pbt_workflow.create_env", side_effect=create_env),
    ):
        workflow = cls.build_from_config(config, jit=True)
    try:
        assert target_cls.step is original_target_step
        assert workflow.mesh.axis_names == (META_POP_AXIS_NAME,)
        assert META_POP_AXIS_NAME != POP_AXIS_NAME
        assert workflow.workflow.dp_axis_name is None
        assert not hasattr(workflow, "rollout_axis_name")
        assert not hasattr(workflow.workflow, "rollout_axis_name")
        with (
            patch("evorl.algorithms.ec.obs_utils.create_env", side_effect=create_env),
            jax.set_mesh(workflow.mesh),
        ):
            state = jax.jit(workflow.setup)(jax.random.key(0))
            if target == "paramopenes":
                # VMA inference must not reduce statistics over META_POP.
                np.testing.assert_array_equal(
                    state.pop_workflow_state.agent_state.obs_preprocessor_state.count,
                    [4] * config.pop_size,
                )

            @jax.shard_map(
                in_specs=jax.P(META_POP_AXIS_NAME),
                out_specs=jax.P(META_POP_AXIS_NAME),
            )
            def train_local_members(states):
                return jax.vmap(workflow.workflow.step)(states)

            executable = (
                jax.jit(train_local_members).lower(state.pop_workflow_state).compile()
            )
            hlo = executable.as_text()
            assert not compiled_collectives(hlo)
            _, trained = executable(state.pop_workflow_state)
            if target in ("ppo", "paramppo"):
                # One population member must match its independent local update.
                member_index = jnp.array([0])
                member = jax.tree.map(
                    lambda x: x.at[member_index].get(out_sharding=jax.P())[0],
                    state.pop_workflow_state,
                )
                _, expected = workflow.workflow.step(member)
                for actual, reference in zip(
                    jax.tree.leaves(trained.agent_state.model_state),
                    jax.tree.leaves(expected.agent_state.model_state),
                    strict=True,
                ):
                    np.testing.assert_allclose(
                        actual.at[member_index].get(out_sharding=jax.P())[0],
                        reference,
                        rtol=1e-5,
                        atol=1e-6,
                    )
            _, state = workflow.step(state)
            metrics, state = workflow.evaluate(state)
            jax.block_until_ready(state)
            assert np.isfinite(metrics.pop_episode_return).all()
            assert state.pop_workflow_state.key.sharding.spec == jax.P(
                META_POP_AXIS_NAME
            )
    finally:
        workflow.close()


@pytest.mark.parametrize("nested", [False, True])
def test_learning_rate_mutation_restores_nnx_optimizer_and_preserves_input(nested, jit):
    """Mutated learning rates affect updates inside a functional NNX snapshot."""
    tx = optax.inject_hyperparams(optax.sgd)(learning_rate=0.01)
    model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    state = State(opt_state=nnx.unpack(optimizer))
    workflow = object.__new__(PBTWorkflow)
    grads = jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param))
    model_state = nnx.unpack(model)

    def mutate_and_update(rate):
        updated = workflow.apply_hyperparams_to_workflow_state(
            state, PyTreeDict(lr=rate)
        )
        local_optimizer = nnx.merge(updated.opt_state, copy=True)
        local_model = nnx.merge(model_state, copy=True)
        local_optimizer.update(local_model, grads)
        return nnx.state(local_model, nnx.Param), nnx.unpack(local_optimizer)

    rates = jnp.array([[0.1, 0.2], [0.3, 0.4]]) if nested else jnp.float32(0.1)
    fn = jax.vmap(jax.vmap(mutate_and_update)) if nested else mutate_and_update
    params, next_opt = (jax.jit(fn) if jit else fn)(rates)
    for actual, original in zip(
        jax.tree.leaves(params),
        jax.tree.leaves(nnx.state(model, nnx.Param)),
        strict=True,
    ):
        expected = original - rates.reshape((*rates.shape, *([1] * original.ndim)))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        nnx.merge(state.opt_state, copy=True).opt_state.hyperparams["learning_rate"][
            ...
        ],
        0.01,
    )
    assert int(nnx.merge(state.opt_state, copy=True).step[...]) == 0
    np.testing.assert_array_equal(
        nnx.merge(next_opt, copy=True).step[...], jnp.ones_like(rates, dtype=jnp.uint32)
    )


@pytest.mark.parametrize("nested", [False, True])
def test_paramsac_trains_with_shared_readonly_buffer_and_native_optimizers(nested):

    env = ToyEnv(2)
    agent = build_agent(
        ParamSACAgent,
        env.obs_space,
        env.action_space,
        actor_hidden_layer_sizes=(4,),
        critic_hidden_layer_sizes=(4,),
    )
    config = OmegaConf.merge(
        OmegaConf.load(Path(__file__).resolve().parents[2] / "configs/config.yaml"),
        {
            "num_envs": 2,
            "rollout_length": 2,
            "actor_update_interval": 2,
            "num_updates_per_iter": 1,
            "tau": 0.1,
        },
    )
    buffer = ReplayBuffer(capacity=16, sample_batch_size=4)
    workflow = ParamSACWorkflow(env, agent, optax.adam(1e-3), None, buffer, config)
    rb_state = workflow._setup_replaybuffer(jax.random.key(1))
    samples = jax.tree.map(
        lambda x: jnp.ones((8, *x.shape[1:]), x.dtype), rb_state.data
    )
    rb_state = buffer.add(rb_state, samples)

    def member(key):
        state = workflow.setup(key).replace(replay_buffer_state=rb_state)
        metric, updated = workflow.step(state)
        return state, metric, updated

    key = jax.random.key(2)
    initial, metric, updated = jax.jit(member)(key)
    assert np.isfinite(metric.actor_loss) and np.isfinite(metric.critic_loss)
    assert int(nnx.merge(updated.opt_state.actor, copy=True).step[...]) == 1
    assert int(nnx.merge(updated.opt_state.critic, copy=True).step[...]) == 2
    initial_models = nnx.merge(initial.agent_state.model_state, copy=True)
    updated_models = nnx.merge(updated.agent_state.model_state, copy=True)
    np.testing.assert_array_equal(
        updated_models.log_alpha[...], initial_models.log_alpha[...]
    )

    pbt = object.__new__(PBTParamSACWorkflow)
    hyperparams = PyTreeDict(
        log_alpha=jnp.float32(-2),
        discount_g=jnp.float32(3),
        actor_loss_weight=jnp.float32(0.5),
        critic_loss_weight=jnp.float32(2),
    )
    adapted = jax.jit(pbt.apply_hyperparams_to_workflow_state)(initial, hyperparams)
    adapted_models = nnx.merge(adapted.agent_state.model_state, copy=True)
    np.testing.assert_array_equal(adapted_models.log_alpha[...], -2)
    np.testing.assert_array_equal(adapted.agent_state.extra_state.discount_g, 3)
    np.testing.assert_array_equal(adapted.hp_state.actor_loss_weight, 0.5)
    np.testing.assert_allclose(
        initial_models.log_alpha[...], jnp.log(jnp.float32(agent.init_alpha)), atol=1e-6
    )
    for target, original, online in zip(
        jax.tree.leaves(nnx.state(updated_models.target_critic, nnx.Param)),
        jax.tree.leaves(nnx.state(initial_models.target_critic, nnx.Param)),
        jax.tree.leaves(nnx.state(updated_models.critic, nnx.Param)),
        strict=True,
    ):
        np.testing.assert_allclose(target, 0.9 * original + 0.1 * online, atol=1e-6)
    for before, after in zip(
        jax.tree.leaves(rb_state),
        jax.tree.leaves(updated.replay_buffer_state),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)
    if nested:
        keys = jax.random.split(key, 6).reshape(2, 3)
        _, _, population = jax.jit(jax.vmap(jax.vmap(member)))(keys)
        _, _, expected = jax.jit(member)(keys[0, 0])
        for actual, reference in zip(
            jax.tree.leaves(population.agent_state.model_state),
            jax.tree.leaves(expected.agent_state.model_state),
            strict=True,
        ):
            np.testing.assert_allclose(actual[0, 0], reference, atol=1e-6)


@pytest.mark.parametrize("num_devices", [1, 2])
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "private"])
def test_pbt_sac_replay_ownership_and_counts_survive_member_replacement(
    num_devices, jit, shared, tmp_path
):
    """Member replacement preserves replay ownership and global sample history."""
    root = Path(__file__).resolve().parents[2]
    common = OmegaConf.load(root / "configs/config.yaml")
    target_config = OmegaConf.merge(
        common,
        OmegaConf.load(root / "configs/agent/meta/paramsac.yaml"),
        dict(
            num_envs=2,
            rollout_length=2,
            num_eval_envs=2,
            eval_episodes=2,
            batch_size=4,
            replay_buffer_capacity=64,
            normalize_obs=True,
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
    )
    config = OmegaConf.merge(
        common,
        OmegaConf.load(root / "configs/agent/meta/pbt-paramsac.yaml"),
        dict(
            target_workflow=target_config,
            env=dict(max_episode_steps=2),
            pop_size=4,
            random_timesteps=4,
            shared_replay_buffer=shared,
            workflow_steps_per_iter=2,
            warmup_steps=0,
            bottom_ratio=0.25,
            top_ratio=0.25,
            num_eval_envs=2,
            eval_episodes=2,
        ),
    )
    member_cls = type("CountingParamSAC", (ParamSACWorkflow,), {})
    workflow_cls = type("CountingPBTSAC", (PBTParamSACWorkflow,), {})

    def create_env(*args, **kwargs):
        return ToyEnv(kwargs["parallel"])

    with (
        patch("hydra.utils.get_class", return_value=member_cls),
        patch(
            "evorl.algorithms.meta.pbt_sac.param_sac.create_env", side_effect=create_env
        ),
        patch("evorl.algorithms.meta.pbt_workflow.create_env", side_effect=create_env),
    ):
        workflow = workflow_cls.build_from_config(
            config, multi_device=num_devices > 1, jit=jit
        )
    if len(jax.local_devices()) < num_devices:
        workflow.close()
        pytest.skip("Requires two devices")
    assert workflow.num_devices == num_devices
    try:
        with jax.set_mesh(workflow.mesh):
            # Force a known parent copy, independent of random policy fitnesses.
            selection = (
                jax.device_put(jnp.array([3]), jax.P()),
                jax.device_put(jnp.array([0]), jax.P()),
            )
            with patch(
                "evorl.algorithms.meta.pbt_workflow.select", return_value=selection
            ):
                state = workflow.setup(jax.random.key(23))
                warmup = 4 if shared else 16
                np.testing.assert_allclose(
                    state.metrics.sampled_timesteps_m, warmup / 1e6
                )
                np.testing.assert_array_equal(
                    state.pop_workflow_state.metrics.sampled_timesteps, [0] * 4
                )
                replay = state.replay_buffer_state
                assert state.pop_workflow_state.replay_buffer_state is None
                assert replay.data.obs.shape == ((64, 3) if shared else (4, 64, 3))
                spec = jax.P() if shared else jax.P(META_POP_AXIS_NAME)
                assert replay.data.obs.sharding.is_equivalent_to(
                    jax.sharding.NamedSharding(workflow.mesh, spec),
                    replay.data.obs.ndim,
                )
                np.testing.assert_array_equal(
                    replay.buffer_size, 4 if shared else [4] * 4
                )
                np.testing.assert_array_equal(
                    state.pop_workflow_state.agent_state.obs_preprocessor_state.count,
                    [4] * 4,
                )
                warmup_obs = np.asarray(
                    replay.data.obs[:4] if shared else replay.data.obs[:, :4]
                )
                if not shared:
                    # Warmup data and normalization belong to each member.
                    assert not np.array_equal(warmup_obs[0], warmup_obs[1])
                    np.testing.assert_allclose(
                        state.pop_workflow_state.agent_state.obs_preprocessor_state.mean,
                        warmup_obs.mean(axis=1),
                        atol=1e-6,
                    )
                else:
                    np.testing.assert_allclose(
                        state.pop_workflow_state.agent_state.obs_preprocessor_state.mean,
                        np.broadcast_to(warmup_obs.mean(axis=0), (4, 3)),
                        atol=1e-6,
                    )

                # Record each member's initial observations. ToyEnv increments
                # them by 0.1 per step, making foreign replay writes detectable.
                initial_obs = np.asarray(state.pop_workflow_state.env_state.obs)
                first_metrics, first = workflow.step(state)
                keys = np.asarray(jax.random.key_data(first.pop_workflow_state.key))
                assert not np.array_equal(keys[0], keys[3])
                expected_keys = state.pop_workflow_state.key
                for _ in range(config.workflow_steps_per_iter):
                    expected_keys = jax.vmap(lambda key: jax.random.split(key, 3)[0])(
                        expected_keys
                    )
                expected_keys = jax.vmap(lambda key: jax.random.split(key)[0])(
                    expected_keys
                )
                np.testing.assert_array_equal(keys, jax.random.key_data(expected_keys))
                assert np.isfinite(first_metrics.pop_train_metrics.actor_loss).all()
                assert np.isfinite(first_metrics.pop_train_metrics.critic_loss).all()
                np.testing.assert_allclose(
                    first.metrics.sampled_timesteps_m, (warmup + 32) / 1e6
                )
                np.testing.assert_array_equal(
                    first.pop_workflow_state.metrics.sampled_timesteps, [8] * 4
                )
                np.testing.assert_array_equal(
                    first_metrics.rb_size, 36 if shared else [12] * 4
                )
                np.testing.assert_array_equal(
                    first.replay_buffer_state.current_index, 36 if shared else [12] * 4
                )
                if not shared:
                    for offset in range(4):
                        np.testing.assert_allclose(
                            first.replay_buffer_state.data.obs[
                                :, 4 + 2 * offset : 6 + 2 * offset
                            ],
                            initial_obs + 0.1 * offset,
                            atol=1e-6,
                        )
                    # The forced copy 3 -> 0 changes member 0's model/env state,
                    # while its replay still contains its own collection history.
                    member_obs = np.asarray(first.pop_workflow_state.env_state.obs)
                    np.testing.assert_array_equal(member_obs[0], member_obs[3])
                    np.testing.assert_array_equal(
                        first.replay_buffer_state.data.obs[:, :4],
                        replay.data.obs[:, :4],
                    )

                # Resume from unequal member histories. Copying member 3 into 0
                # must not count the parent's historical samples a second time.
                history = jax.device_put(
                    jnp.array([10, 20, 30, 40], dtype=jnp.uint32),
                    jax.P(META_POP_AXIS_NAME),
                )
                members = first.pop_workflow_state
                resumed = first.replace(
                    metrics=first.metrics.replace(
                        sampled_timesteps_m=jnp.float32(136e-6)
                    ),
                    pop_workflow_state=members.replace(
                        metrics=members.metrics.replace(
                            sampled_timesteps=members.metrics.sampled_timesteps
                            + history
                        )
                    ),
                )
                if jit:
                    path = tmp_path / "checkpoint"
                    save(path, resumed)
                    restored = load(path, resumed)
                    _, second = workflow.step(restored)
                    _, expected_second = workflow.step(resumed)
                    for actual, expected in zip(
                        jax.tree.leaves(second.replace(replay_buffer_state=None)),
                        jax.tree.leaves(
                            expected_second.replace(replay_buffer_state=None)
                        ),
                        strict=True,
                    ):
                        if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
                            actual, expected = (
                                jax.random.key_data(actual),
                                jax.random.key_data(expected),
                            )
                        np.testing.assert_allclose(
                            actual, expected, rtol=1e-6, atol=1e-6
                        )
                    replay = second.replay_buffer_state
                    expected_replay = expected_second.replay_buffer_state
                    np.testing.assert_array_equal(
                        replay.current_index, expected_replay.current_index
                    )
                    np.testing.assert_array_equal(
                        replay.buffer_size, expected_replay.buffer_size
                    )
                    written = min(4 + 2 * (32 if shared else 8), 64)
                    for actual, expected in zip(
                        jax.tree.leaves(replay.data),
                        jax.tree.leaves(expected_replay.data),
                        strict=True,
                    ):
                        actual, expected = np.asarray(actual), np.asarray(expected)
                        actual = actual[:written] if shared else actual[:, :written]
                        expected = (
                            expected[:written] if shared else expected[:, :written]
                        )
                        np.testing.assert_allclose(
                            actual, expected, rtol=1e-6, atol=1e-6
                        )
                else:
                    _, second = workflow.step(resumed)
                _, third = workflow.step(second)
                np.testing.assert_allclose(second.metrics.sampled_timesteps_m, 168e-6)
                np.testing.assert_allclose(third.metrics.sampled_timesteps_m, 200e-6)
                np.testing.assert_array_equal(
                    second.pop_workflow_state.metrics.sampled_timesteps,
                    [56, 36, 46, 56],
                )
                np.testing.assert_array_equal(
                    third.pop_workflow_state.metrics.sampled_timesteps,
                    [64, 44, 54, 64],
                )
                assert int(third.metrics.iterations) == 3
                np.testing.assert_allclose(
                    state.metrics.sampled_timesteps_m, warmup / 1e6
                )
                np.testing.assert_array_equal(
                    resumed.pop_workflow_state.metrics.sampled_timesteps,
                    [18, 28, 38, 48],
                )
    finally:
        workflow.close()
