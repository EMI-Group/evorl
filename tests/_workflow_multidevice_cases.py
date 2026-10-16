"""Exercise actual RL/EC methods through the class-level shard_map boundary."""

from pathlib import Path
import re
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import get_class
from omegaconf import OmegaConf

from evorl.envs import Box, Discrete, Env, EnvState
from evorl.types import PyTreeDict
from evorl.workflows import OnPolicyWorkflow


class ToyEnv(Env):
    def __init__(self, num_envs, discrete=False):
        self.num_envs = num_envs
        self.discrete = discrete

    @property
    def obs_space(self):
        return Box(low=-jnp.ones(3), high=jnp.ones(3))

    @property
    def action_space(self):
        if self.discrete:
            return Discrete(n=2)
        return Box(low=-jnp.ones(1), high=jnp.ones(1))

    def reset(self, key):
        obs = jax.random.uniform(key, (self.num_envs, 3))
        zeros = jnp.zeros(self.num_envs)
        return EnvState(
            env_state=jnp.zeros(self.num_envs, dtype=jnp.int32),
            obs=obs,
            reward=zeros,
            done=zeros,
            info=PyTreeDict(
                autoreset=zeros, episode_return=zeros, termination=zeros, ori_obs=obs
            ),
        )

    def step(self, state, action):
        count = state.env_state + 1
        done = (count % 2 == 0).astype(jnp.float32)
        reward = 1 - (
            action.astype(jnp.float32) if self.discrete else jnp.square(action).sum(-1)
        )
        obs = state.obs + 0.1
        return state.replace(
            env_state=count,
            obs=obs,
            reward=reward,
            done=done,
            info=state.info.replace(
                autoreset=jnp.zeros_like(done),
                episode_return=2 * reward,
                termination=done,
                ori_obs=obs,
            ),
        )


def run_case(agent, jit):
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / f"configs/agent/{agent}.yaml"),
        dict(
            env=dict(max_episode_steps=2),
            num_envs=4,
            num_eval_envs=4,
            eval_episodes=4,
            normalize_obs=True,
            rollout_length=2,
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
    )
    if agent == "ppo":
        config.minibatch_size = 8
        config.reuse_rollout_epochs = 1
    elif agent == "sac":
        config.rollout_length = 1
        config.batch_size = 4
        config.replay_buffer_capacity = 16
        config.random_timesteps = 4
        config.learning_start_timesteps = 4
        config.num_updates_per_iter = 1
        config.fold_iters = 2
        config.adaptive_alpha = True
    elif agent == "dqn":
        config.batch_size = 2
        config.replay_buffer_capacity = 16
        config.random_timesteps = 4
        config.learning_start_timesteps = 4
        config.agent_network.q_hidden_layer_sizes = [4]
    elif agent in ("td3", "ddpg", "contrib/td3-v3"):
        config.batch_size = 2
        config.replay_buffer_capacity = 16
        config.random_timesteps = 4
        config.learning_start_timesteps = 4
        config.num_updates_per_iter = 1
        config.rollout_length = 2
        if agent != "ddpg":
            config.actor_update_interval = 1
        if agent == "contrib/td3-v3":
            config.normalize_obs = False
    elif agent == "td7":
        config.rollout_episodes = 4
        config.batch_size = 2
        config.replay_buffer_capacity = 16
        config.random_timesteps = 4
        config.learning_start_timesteps = 4
        config.num_updates_per_iter = 2
        config.target_update_rate = 1
        config.policy_freq = 1
        config.agent_network = dict(
            zs_dim=4,
            zsa_dim=4,
            f_layer_sizes=[4],
            g_layer_sizes=[4],
            state_emb_dim=4,
            state_action_emb_dim=4,
            critic_hidden_layer_sizes=[4],
            actor_hidden_layer_sizes=[4],
        )
    elif agent == "ec/openes":
        config.pop_size = 4
        config.num_envs = 1
        config.episodes_for_fitness = 1
        config.random_timesteps = 4

    base = get_class(config.workflow_cls)
    original_step, original_evaluate = base.step, base.evaluate
    cls = type(f"Distributed{base.__name__}", (base,), {})
    local_cls = type(f"Local{base.__name__}", (base,), {})

    def create_env(*args, **kwargs):
        return ToyEnv(kwargs["parallel"], discrete=agent == "dqn")

    with patch(f"{base.__module__}.create_env", side_effect=create_env):
        workflow = cls.build_from_config(config, multi_device=True, jit=jit)
        local = local_cls.build_from_config(config, multi_device=False, jit=False)
    if agent != "ec/openes":
        assert local.config.eval_episodes == config.eval_episodes
        assert (
            workflow.config.eval_episodes * jax.device_count() == config.eval_episodes
        )
        assert (
            workflow.evaluator.env.num_envs * jax.device_count() == config.num_eval_envs
        )
    else:
        assert local.config.random_timesteps == config.random_timesteps
        assert workflow.config.random_timesteps == config.random_timesteps // 2
        assert workflow.config.eval_episodes == config.eval_episodes // 2
        assert workflow.evaluator.env.num_envs == config.num_eval_envs // 2
    if agent == "dqn":
        for update in (0, 1, 32, 1024):
            np.testing.assert_array_equal(
                workflow.epsilon_scheduler(update), local.epsilon_scheduler(update)
            )
    if agent == "random":
        for requested_episodes in (1, 5):
            rounded_config = OmegaConf.create(
                dict(num_eval_envs=4, eval_episodes=requested_episodes)
            )
            base._rescale_config(rounded_config)
            assert rounded_config.num_eval_envs == 2
            assert (
                rounded_config.eval_episodes * 2 == ((requested_episodes + 3) // 4) * 4
            )
    if agent in ("sac", "dqn", "td3", "ddpg", "contrib/td3-v3", "td7"):
        assert local.config.batch_size == config.batch_size
        assert workflow.config.batch_size * jax.device_count() == config.batch_size
        assert workflow.replay_buffer.sample_batch_size == workflow.config.batch_size
    if agent == "td7":
        assert local.config.rollout_episodes == config.rollout_episodes
        assert (
            workflow.config.rollout_episodes * jax.device_count()
            == config.rollout_episodes
        )
    assert base.step is original_step and base.evaluate is original_evaluate
    assert local.step.__func__ is original_step
    assert local.mesh is None and not local.is_multi_device
    assert "step" not in vars(workflow) and "evaluate" not in vars(workflow)
    if agent != "ec/openes":
        assert hasattr(cls.setup, "lower") == jit
    assert hasattr(cls.step, "lower") == jit
    assert hasattr(cls.evaluate, "lower") == jit
    if agent == "sac":
        assert hasattr(cls._multi_steps, "lower") == jit
        assert hasattr(cls._postsetup_replaybuffer, "lower") == jit
    if agent == "ec/openes":
        assert hasattr(cls._postsetup, "lower") == jit
    local.close()

    try:
        with (
            patch("evorl.algorithms.ec.obs_utils.create_env", side_effect=create_env),
            jax.set_mesh(workflow.mesh),
        ):
            state = workflow.setup(jax.random.key(12))
            if agent == "ec/openes":
                assert (
                    state.agent_state.obs_preprocessor_state.count
                    == config.random_timesteps
                )
            if agent == "td7":
                rollout_key = jax.random.split(state.key, 3)[1]

                @jax.shard_map(
                    in_specs=(jax.P(), jax.P()),
                    out_specs=jax.P(workflow.dp_axis_name),
                )
                def local_checkpoint_perf(agent_state, key):
                    key = jax.random.fold_in(
                        key, jax.lax.axis_index(workflow.dp_axis_name)
                    )
                    episode_metrics, _ = workflow.collector.rollout(
                        agent_state, key, workflow.config.rollout_episodes
                    )
                    return jnp.min(episode_metrics.episode_returns)[None]

                expected_best_perf = jnp.min(
                    jax.jit(local_checkpoint_perf)(state.agent_state, rollout_key)
                )
            if agent in (
                "ppo",
                "sac",
                "dqn",
                "td3",
                "ddpg",
                "contrib/td3-v3",
                "td7",
            ):
                # The global environment batch has no leading device dimension.
                assert state.env_state.obs.shape == (config.num_envs, 3)
                observations = np.asarray(state.env_state.obs)
                assert not np.array_equal(observations[:2], observations[2:])
                for value in jax.tree.leaves(state.env_state):
                    assert value.shape[0] == config.num_envs
                    assert value.sharding.spec[0] == workflow.dp_axis_name
                    assert all(axis is None for axis in value.sharding.spec[1:])
                    assert all(
                        shard.data.shape[0] == workflow.config.num_envs
                        for shard in value.addressable_shards
                    )
            if jit:
                allowed = {"all-reduce"}
                if agent == "ec/openes":
                    allowed.add("all-gather")
                if agent == "random":
                    allowed = set()
                hlo = cls.step.lower(workflow, state).compile().as_text()
                collective_pattern = r"\b(all-reduce|all-gather|all-to-all|collective-permute|reduce-scatter|collective-broadcast)(?:-start|-done)?\("
                collectives = set(re.findall(collective_pattern, hlo))
                assert collectives <= allowed, (agent, collectives)
                if agent in (
                    "ppo",
                    "sac",
                    "dqn",
                    "td3",
                    "ddpg",
                    "contrib/td3-v3",
                    "td7",
                ):
                    assert "all-reduce" in collectives
                print(
                    f"{agent} compiled step collectives: {sorted(collectives)}",
                    flush=True,
                )
                if agent == "ppo":
                    # A computation with no collectives must stay local even when
                    # the real Workflow state crosses the shard_map boundary.
                    class LocalOnlyWorkflow(OnPolicyWorkflow):
                        def step(self, state):
                            return jnp.float32(0), state.replace(
                                env_state=state.env_state.replace(
                                    obs=state.env_state.obs * 2
                                )
                            )

                    LocalOnlyWorkflow.enable_shmap()
                    LocalOnlyWorkflow.enable_jit()
                    probe = object.__new__(LocalOnlyWorkflow)
                    executable = LocalOnlyWorkflow.step.lower(probe, state).compile()
                    assert not re.search(collective_pattern, executable.as_text())
                    _, probe_state = probe.step(state)
                    np.testing.assert_allclose(
                        probe_state.env_state.obs, np.asarray(state.env_state.obs) * 2
                    )
                    print(
                        "local-only workflow boundary: no collective communication",
                        flush=True,
                    )
            metrics, updated = workflow.step(state)
            eval_metrics, evaluated = workflow.evaluate(updated)
            jax.block_until_ready(evaluated)
            if agent == "ec/openes":
                assert evaluated.key.shape == (2,)
                assert evaluated.key.sharding.spec == jax.P(workflow.dp_axis_name)
            else:
                assert evaluated.key.shape == ()
                assert evaluated.key.sharding.is_fully_replicated
                expected_key = state.key
                if agent != "random":
                    expected_key = jax.random.split(expected_key, 3)[0]
                expected_key = jax.random.split(expected_key, 2)[0]
                np.testing.assert_array_equal(
                    jax.random.key_data(evaluated.key),
                    jax.random.key_data(expected_key),
                )
                for shard in evaluated.key.addressable_shards:
                    np.testing.assert_array_equal(
                        jax.random.key_data(shard.data),
                        jax.random.key_data(evaluated.key),
                    )
            assert jax.tree.structure(evaluated) == jax.tree.structure(state)
            for before, after in zip(
                jax.tree.leaves(state), jax.tree.leaves(evaluated), strict=True
            ):
                assert before.shape == after.shape
                assert before.sharding.is_equivalent_to(after.sharding, before.ndim), (
                    before.sharding,
                    after.sharding,
                )
            if agent in ("ppo", "sac", "dqn", "td3", "ddpg", "contrib/td3-v3"):
                np.testing.assert_allclose(
                    evaluated.env_state.obs,
                    np.asarray(state.env_state.obs)
                    + 0.1 * workflow.config.rollout_length,
                    rtol=1e-6,
                )
            for value in jax.tree.leaves(
                (evaluated.agent_state.params, metrics, eval_metrics)
            ):
                assert value.sharding.is_fully_replicated
                assert np.isfinite(np.asarray(value)).all()
                for shard in value.addressable_shards:
                    np.testing.assert_allclose(shard.data, np.asarray(value), rtol=1e-6)
            if agent == "sac":
                assert evaluated.replay_buffer_state.data.obs.shape == (
                    config.replay_buffer_capacity,
                    3,
                )
                assert evaluated.replay_buffer_state.current_index.shape == (2,)
                assert all(
                    shard.data.shape == (workflow.replay_buffer.capacity, 3)
                    for shard in evaluated.replay_buffer_state.data.obs.addressable_shards
                )
                assert (
                    evaluated.replay_buffer_state.data.obs.sharding.spec[0]
                    == workflow.dp_axis_name
                )
                _, folded = workflow._multi_steps(evaluated)
                jax.block_until_ready(folded)
                assert (
                    int(folded.metrics.iterations - evaluated.metrics.iterations)
                    == config.fold_iters
                )
            if agent == "td7":
                bounds = evaluated.agent_state.extra_state
                assert evaluated.metrics.sampled_episodes == config.rollout_episodes
                for value in (
                    bounds.max_q,
                    bounds.min_q,
                    bounds.max_target,
                    bounds.min_target,
                    bounds.best_perf,
                ):
                    assert value.sharding.is_fully_replicated
                    for shard in value.addressable_shards:
                        np.testing.assert_allclose(shard.data, np.asarray(value))
                assert bounds.min_q <= bounds.max_q
                assert bounds.min_target == bounds.min_q
                assert bounds.max_target == bounds.max_q
                np.testing.assert_allclose(bounds.best_perf, expected_best_perf)
            if agent == "ec/openes":
                np.testing.assert_array_equal(evaluated.distributed_info.rank, [0, 1])
    finally:
        workflow.close()

    if agent == "ec/openes" and jit:
        rounded_config = OmegaConf.create(OmegaConf.to_container(config))
        rounded_config.num_envs = 2
        rounded_config.random_timesteps = 5
        rounded_cls = type("RoundedOpenESWorkflow", (base,), {})
        with patch(f"{base.__module__}.create_env", side_effect=create_env):
            rounded_workflow = rounded_cls.build_from_config(
                rounded_config, multi_device=True, jit=True
            )
        try:
            with (
                patch(
                    "evorl.algorithms.ec.obs_utils.create_env", side_effect=create_env
                ),
                jax.set_mesh(rounded_workflow.mesh),
            ):
                rounded_state = rounded_workflow.setup(jax.random.key(13))
                # Single-device warmup collects 6; whole per-device batches need 8.
                assert rounded_workflow.config.random_timesteps == 4
                assert rounded_state.agent_state.obs_preprocessor_state.count == 8
        finally:
            rounded_workflow.close()
    print(f"{agent}, jit={jit}: setup/step/evaluate PASS", flush=True)


def check_invalid_multidevice_configs():
    root = Path(__file__).resolve().parents[1]
    for agent, field, value in (
        ("ppo", "num_envs", 1),
        ("ppo", "minibatch_size", 1),
        ("sac", "replay_buffer_capacity", 1),
        ("sac", "random_timesteps", 1),
        ("ec/openes", "pop_size", 3),
        ("ec/openes", "num_eval_envs", 1),
        ("ec/openes", "random_timesteps", 0),
    ):
        config = OmegaConf.merge(
            OmegaConf.load(root / "configs/config.yaml"),
            OmegaConf.load(root / f"configs/agent/{agent}.yaml"),
            {"num_envs": 4, "num_eval_envs": 4, "eval_episodes": 4},
            {field: value},
        )
        workflow_cls = get_class(config.workflow_cls)
        try:
            workflow_cls._rescale_config(config)
        except ValueError as exc:
            assert field in str(exc), (agent, field, exc)
        else:
            raise AssertionError(f"{agent}: {field}={value} should fail before build")


if __name__ == "__main__":
    assert jax.device_count() == 2
    check_invalid_multidevice_configs()
    jit = bool(int(sys.argv[1]))
    for agent in (
        "ppo",
        "sac",
        "dqn",
        "td3",
        "ddpg",
        "contrib/td3-v3",
        "td7",
        "ec/openes",
        "random",
    ):
        run_case(agent, jit)
