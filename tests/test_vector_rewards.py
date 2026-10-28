"""Scalar and vector rewards through wrappers, rollout and evaluation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from evorl.envs import Box, Env, EnvState, AutoresetMode, create_env
from evorl.envs.wrappers import (
    ActionRepeatWrapper,
    EpisodeWrapper,
    FastVmapAutoResetWrapper,
    BraxMultiObjectiveRewardWrapper,
    MujocoMultiObjectiveRewardWrapper,
    OneEpisodeWrapper,
    RewardScaleWrapper,
    SparseRewardWrapper,
    VmapAutoResetWrapper,
    VmapEnvPoolAutoResetWrapper,
    VmapWrapper,
    get_wrapper,
)
from evorl.evaluators import Evaluator, EpisodeCollector
from evorl.rollout import (
    fast_eval_rollout_episode,
    host_fast_eval_rollout_episode,
    rollout,
)
from evorl.types import PyTreeDict
from evorl.utils.rl_toolkits import (
    average_episode_discount_return,
    compute_discount_return,
)
from evorl.workflows.ec_workflow import MultiObjectiveECWorkflowTemplate
from tests.helpers.sharding import compiled_collectives


REWARD_KEYS = [None, ("progress",), ("progress", "control")]
REWARD_IDS = ["scalar", "one_objective", "two_objectives"]


class MetricEnv(Env):
    """Expose two known reward terms and episodes lasting one to three steps."""

    @property
    def obs_space(self):
        return Box(low=jnp.zeros(1), high=jnp.full((1,), 10.0))

    @property
    def action_space(self):
        return Box(low=-jnp.ones(1), high=jnp.ones(1))

    def reset(self, key):
        zero = jnp.float32(0)
        return EnvState(
            env_state=PyTreeDict(
                count=jnp.int32(0),
                horizon=jax.random.randint(key, (), 1, 4),
                native_reward=zero,
            ),
            obs=jnp.zeros(1),
            reward=zero,
            done=zero,
            info=PyTreeDict(metrics=PyTreeDict(progress=zero, control=zero)),
        )

    def step(self, state, action):
        count = state.env_state.count + 1
        progress, control = count.astype(jnp.float32), jnp.float32(-2)
        reward = progress + control
        return state.replace(
            env_state=state.env_state.replace(count=count, native_reward=reward),
            obs=jnp.array([progress]),
            reward=reward,
            done=(count >= state.env_state.horizon).astype(jnp.float32),
            info=state.info.replace(
                metrics=PyTreeDict(progress=progress, control=control)
            ),
        )


def make_metric_env(reward_keys):
    env = MetricEnv()
    if reward_keys is not None:
        env = BraxMultiObjectiveRewardWrapper(env, reward_keys)
    return env


def action_fn(agent_state, batch, key):
    return jnp.zeros_like(batch.obs), PyTreeDict()


def expected_returns(lengths, reward_keys, discount=1.0):
    """Sum known reward terms only through each episode's terminal step."""
    steps = np.arange(1, 6)
    weights = (steps <= np.asarray(lengths)[..., None]) * discount ** (steps - 1)
    progress = (weights * steps).sum(-1)
    control = -2 * weights.sum(-1)
    if reward_keys is None:
        return progress + control
    terms = {"progress": progress, "control": control}
    return np.stack([terms[name] for name in reward_keys], axis=-1)


@pytest.mark.parametrize("reward_keys", REWARD_KEYS, ids=REWARD_IDS)
@pytest.mark.parametrize("discount", [1.0, 0.5])
def test_evaluator_and_collector_preserve_reward_axes(reward_keys, discount):
    env = VmapWrapper(
        OneEpisodeWrapper(make_metric_env(reward_keys), episode_length=5),
        num_envs=3,
        vmap_step=True,
    )
    evaluator = Evaluator(
        env=env, action_fn=action_fn, max_episode_steps=5, discount=discount
    )
    collector = EpisodeCollector(
        env=env, action_fn=action_fn, max_episode_steps=5, discount=discount
    )
    keys = jax.random.split(jax.random.key(2), (2, 2))
    evaluate = jax.jit(jax.vmap(jax.vmap(lambda k: evaluator.evaluate(None, k, 4))))
    collect = jax.jit(jax.vmap(jax.vmap(lambda k: collector.rollout(None, k, 4))))
    metrics = evaluate(keys)
    collected, trajectory = collect(keys)
    reward_shape = () if reward_keys is None else (len(reward_keys),)
    assert metrics.episode_return.shape == (2, 2, 6, *reward_shape)
    assert metrics.episode_length.shape == (2, 2, 6)
    assert collected.episode_return.shape == metrics.episode_return.shape
    assert trajectory.rewards.shape == (2, 2, 5, 6, *reward_shape)
    assert trajectory.dones.shape == (2, 2, 5, 6)
    for result in (metrics, collected):
        np.testing.assert_allclose(
            result.episode_return,
            expected_returns(result.episode_length, reward_keys, discount),
        )


@pytest.mark.parametrize("reward_keys", REWARD_KEYS, ids=REWARD_IDS)
@pytest.mark.parametrize("rollout_length", [2, 5], ids=["step_limit", "early_stop"])
@pytest.mark.parametrize(
    "rollout_fn",
    [fast_eval_rollout_episode, host_fast_eval_rollout_episode],
    ids=["jax", "host"],
)
def test_nested_fast_evaluation_masks_each_environment(
    reward_keys, rollout_length, rollout_fn
):
    env = VmapWrapper(
        OneEpisodeWrapper(make_metric_env(reward_keys), episode_length=5),
        num_envs=3,
        vmap_step=True,
    )

    def evaluate(key):
        state = env.reset(key)
        metrics, _ = rollout_fn(env.step, action_fn, state, None, key, rollout_length)
        return metrics, state.env_state.horizon

    keys = jax.random.split(jax.random.key(3), (2, 2))
    metrics, horizons = jax.jit(jax.vmap(jax.vmap(evaluate)))(keys)
    assert np.unique(np.asarray(horizons)).size > 1
    lengths = np.minimum(horizons, rollout_length)
    np.testing.assert_array_equal(metrics.episode_length, lengths)
    np.testing.assert_allclose(
        metrics.episode_return, expected_returns(lengths, reward_keys)
    )


@pytest.mark.parametrize("reward_keys", REWARD_KEYS, ids=REWARD_IDS)
def test_action_repeat_keeps_reward_objectives_and_terminal_step(reward_keys):
    env = RewardScaleWrapper(make_metric_env(reward_keys), reward_scale=3.0)
    env = ActionRepeatWrapper(OneEpisodeWrapper(env, episode_length=2), action_repeat=3)
    state = env.reset(jax.random.key(0))
    state = state.replace(env_state=state.env_state.replace(horizon=jnp.int32(3)))
    state = jax.jit(env.step)(state, jnp.zeros(1))
    expected = expected_returns(np.array(2), reward_keys)
    np.testing.assert_allclose(state.reward, expected * 3)
    np.testing.assert_allclose(state.info.ori_reward, expected)
    assert state.reward.shape == (() if reward_keys is None else (len(reward_keys),))
    assert state.env_state.native_reward.shape == ()
    assert state.info.steps == 2


@pytest.mark.parametrize("reward_keys", REWARD_KEYS, ids=REWARD_IDS)
@pytest.mark.parametrize("mode", list(AutoresetMode))
def test_episode_wrappers_preserve_terminal_reward_under_nested_vmap(reward_keys, mode):
    env = make_metric_env(reward_keys)
    if mode == AutoresetMode.DISABLED:
        env = VmapWrapper(
            OneEpisodeWrapper(env, episode_length=1, discount=1.0),
            num_envs=3,
            vmap_step=True,
        )
    else:
        wrapper = {
            AutoresetMode.NORMAL: VmapAutoResetWrapper,
            AutoresetMode.FAST: FastVmapAutoResetWrapper,
            AutoresetMode.ENVPOOL: VmapEnvPoolAutoResetWrapper,
        }[mode]
        env = wrapper(EpisodeWrapper(env, episode_length=1, discount=1.0), num_envs=3)

    def run(key):
        state = env.reset(key)
        trajectory, _ = rollout(
            env.step,
            action_fn,
            state,
            None,
            key,
            3,
            env_extra_fields=("episode_return",),
        )
        return trajectory

    keys = jax.random.split(jax.random.key(4), (2, 2))
    trajectory = jax.jit(jax.vmap(jax.vmap(run)))(keys)
    reward_shape = () if reward_keys is None else (len(reward_keys),)
    assert trajectory.rewards.shape == (2, 2, 3, 3, *reward_shape)
    assert trajectory.dones.shape == (2, 2, 3, 3)
    expected = expected_returns(np.ones((2, 2, 3)), reward_keys)
    np.testing.assert_allclose(trajectory.rewards[:, :, 0], expected)
    np.testing.assert_allclose(
        trajectory.extras.env_extras.episode_return[:, :, 0], expected
    )
    if mode == AutoresetMode.ENVPOOL:
        np.testing.assert_array_equal(trajectory.rewards[:, :, 1], 0)
        np.testing.assert_allclose(trajectory.rewards[:, :, 2], expected)
    else:
        np.testing.assert_allclose(trajectory.rewards[:, :, 1], expected)


@pytest.mark.parametrize("reward_keys", REWARD_KEYS, ids=REWARD_IDS)
def test_sparse_rewards_accumulate_each_objective(reward_keys):
    env = SparseRewardWrapper(
        OneEpisodeWrapper(make_metric_env(reward_keys), episode_length=3),
        sparse_length=2,
    )
    state = env.reset(jax.random.key(0))
    state = state.replace(env_state=state.env_state.replace(horizon=jnp.int32(3)))
    step = jax.jit(env.step)
    state = step(state, jnp.zeros(1))
    np.testing.assert_array_equal(state.reward, np.zeros_like(state.reward))
    state = step(state, jnp.zeros(1))
    np.testing.assert_allclose(state.reward, expected_returns(np.array(2), reward_keys))
    state = step(state, jnp.zeros(1))
    np.testing.assert_allclose(
        state.reward,
        expected_returns(np.array(3), reward_keys)
        - expected_returns(np.array(2), reward_keys),
    )


@pytest.mark.parametrize("reward_keys", REWARD_KEYS, ids=REWARD_IDS)
@pytest.mark.parametrize("mode", list(AutoresetMode))
def test_sparse_rewards_flush_on_external_truncation_under_nested_vmap(
    reward_keys, mode
):
    class LongEpisodeEnv(MetricEnv):
        def reset(self, key):
            state = super().reset(key)
            return state.replace(
                env_state=state.env_state.replace(horizon=jnp.int32(10))
            )

    env = LongEpisodeEnv()
    if reward_keys is not None:
        env = BraxMultiObjectiveRewardWrapper(env, reward_keys)
    episode_wrapper = (
        OneEpisodeWrapper if mode == AutoresetMode.DISABLED else EpisodeWrapper
    )
    env = SparseRewardWrapper(
        episode_wrapper(env, episode_length=2, discount=1.0), sparse_length=3
    )
    if mode == AutoresetMode.DISABLED:
        env = VmapWrapper(env, num_envs=2, vmap_step=True)
    else:
        wrapper = {
            AutoresetMode.NORMAL: VmapAutoResetWrapper,
            AutoresetMode.FAST: FastVmapAutoResetWrapper,
            AutoresetMode.ENVPOOL: VmapEnvPoolAutoResetWrapper,
        }[mode]
        env = wrapper(env, num_envs=2)

    def run(key):
        def step(state, _):
            state = env.step(state, jnp.zeros((2, 1)))
            return state, (
                state.reward,
                state.done,
                state._internal.cum_count,
                state._internal.cum_reward,
                state.info.truncation,
                state.info.episode_return,
            )

        return jax.lax.scan(step, env.reset(key), None, length=6)[1]

    keys = jax.random.split(jax.random.key(5), (2, 2))
    rewards, dones, counts, pending, truncations, returns = jax.jit(
        jax.vmap(jax.vmap(run))
    )(keys)
    if mode == AutoresetMode.DISABLED:
        expected_counts = [1, 0, 0, 0, 0, 0]
        expected_dones = [0, 1, 1, 1, 1, 1]
        expected_lengths = [1, 2, 2, 2, 2, 2]
    elif mode == AutoresetMode.ENVPOOL:
        expected_counts = [1, 0, 0, 1, 0, 0]
        expected_dones = [0, 1, 0, 0, 1, 0]
        expected_lengths = [1, 2, 0, 1, 2, 0]
    else:
        expected_counts = [1, 0, 1, 0, 1, 0]
        expected_dones = [0, 1, 0, 1, 0, 1]
        expected_lengths = [1, 2, 1, 2, 1, 2]

    def broadcast_steps(values, shape):
        return np.broadcast_to(
            np.asarray(values).reshape((1, 1, 6, 1, *shape[4:])), shape
        )

    reward_shape = () if reward_keys is None else (len(reward_keys),)
    assert rewards.shape == (2, 2, 6, 2, *reward_shape)
    expected_rewards = expected_returns(np.array(expected_dones) * 2, reward_keys)
    np.testing.assert_allclose(
        rewards, broadcast_steps(expected_rewards, rewards.shape)
    )
    np.testing.assert_array_equal(dones, broadcast_steps(expected_dones, dones.shape))
    np.testing.assert_array_equal(truncations, dones)
    np.testing.assert_array_equal(
        counts, broadcast_steps(expected_counts, counts.shape)
    )
    np.testing.assert_allclose(
        pending,
        broadcast_steps(expected_returns(expected_counts, reward_keys), pending.shape),
    )
    np.testing.assert_allclose(
        returns,
        broadcast_steps(expected_returns(expected_lengths, reward_keys), returns.shape),
    )


@pytest.mark.parametrize("reward_keys", REWARD_KEYS, ids=REWARD_IDS)
def test_episode_wrapper_records_discounted_returns_per_objective(reward_keys):
    env = OneEpisodeWrapper(
        make_metric_env(reward_keys), episode_length=3, discount=0.5
    )
    state = env.reset(jax.random.key(0))
    state = state.replace(env_state=state.env_state.replace(horizon=jnp.int32(3)))
    step = jax.jit(env.step)
    for _ in range(3):
        state = step(state, jnp.zeros(1))
    np.testing.assert_allclose(
        state.info.episode_return, expected_returns(np.array(3), reward_keys, 0.5)
    )
    assert state.info.steps.shape == ()
    assert state.done.shape == ()


@pytest.mark.parametrize("num_envs", [None, 2, 3])
@pytest.mark.parametrize("reward_keys", REWARD_KEYS, ids=REWARD_IDS)
def test_discount_returns_and_episode_statistics_keep_objectives(num_envs, reward_keys):
    lengths = np.array(3) if num_envs is None else np.arange(1, num_envs + 1)
    steps = np.arange(1, 6).reshape((-1,) + (1,) * lengths.ndim)
    progress = np.broadcast_to(steps, (5, *lengths.shape)).astype(np.float32)
    control = np.full_like(progress, -2)
    rewards = (
        progress + control
        if reward_keys is None
        else np.stack(
            [{"progress": progress, "control": control}[name] for name in reward_keys],
            axis=-1,
        )
    )
    dones = (steps >= lengths).astype(np.float32)
    actual = jax.jit(compute_discount_return)(jnp.array(rewards), jnp.array(dones), 0.5)
    np.testing.assert_allclose(actual, expected_returns(lengths, reward_keys, 0.5))

    terminal_returns = jnp.ones_like(rewards)
    expected_shape = () if reward_keys is None else (len(reward_keys),)
    mean_return = jax.jit(average_episode_discount_return)(
        terminal_returns, jnp.array(dones)
    )
    assert mean_return.shape == expected_shape
    np.testing.assert_array_equal(mean_return, jnp.ones(expected_shape))


@pytest.mark.parametrize("reward_keys", [(), "progress"])
def test_reward_wrapper_rejects_invalid_selection(reward_keys):
    with pytest.raises(ValueError, match="nonempty sequence"):
        BraxMultiObjectiveRewardWrapper(MetricEnv(), reward_keys)


def test_reward_wrapper_checks_metric_names_order_and_native_state():
    env = BraxMultiObjectiveRewardWrapper(MetricEnv(), ("control", "progress"))
    state = env.reset(jax.random.key(0))
    np.testing.assert_array_equal(state.reward, [0, 0])
    state = env.step(state, jnp.zeros(1))
    np.testing.assert_array_equal(state.reward, [-2, 1])
    assert state.env_state.native_reward.shape == ()
    with pytest.raises(KeyError, match="missing"):
        BraxMultiObjectiveRewardWrapper(MetricEnv(), ("missing",)).reset(
            jax.random.key(0)
        )


@pytest.mark.parametrize("num_objectives", [1, 2])
def test_multiobjective_fitness_averages_episodes_only(num_objectives):
    returns = jnp.arange(3 * 4 * num_objectives, dtype=jnp.float32).reshape(
        3, 4, num_objectives
    )
    workflow = object.__new__(MultiObjectiveECWorkflowTemplate)
    actual = workflow._metrics_to_fitnesses(PyTreeDict(episode_return=returns))
    np.testing.assert_allclose(actual, np.asarray(returns).mean(axis=1))
    assert actual.shape == (3, num_objectives)


@pytest.mark.parametrize("reward_keys", REWARD_KEYS, ids=REWARD_IDS)
def test_sharded_evaluation_preserves_objectives_without_communication(
    dp_mesh, reward_keys
):
    env = VmapWrapper(
        OneEpisodeWrapper(make_metric_env(reward_keys), episode_length=5),
        num_envs=3,
        vmap_step=True,
    )
    evaluator = Evaluator(env=env, action_fn=action_fn, max_episode_steps=5)
    axis_name = dp_mesh.axis_names[0]

    def evaluate(key):
        key = jax.random.fold_in(key, jax.lax.axis_index(axis_name))
        return evaluator.evaluate(None, key, 3)

    evaluate = jax.jit(
        jax.shard_map(
            evaluate, mesh=dp_mesh, in_specs=jax.P(), out_specs=jax.P(axis_name)
        )
    )
    key = jax.device_put(jax.random.key(1), jax.P())
    compiled = evaluate.lower(key).compile()
    metrics = compiled(key)
    assert metrics.episode_length.shape == (dp_mesh.size * 3,)
    np.testing.assert_allclose(
        metrics.episode_return,
        expected_returns(metrics.episode_length, reward_keys),
    )
    # Shards evaluate independent episodes; gathering them is not part of rollout.
    assert not compiled_collectives(compiled.as_text())


@pytest.mark.integration
@pytest.mark.parametrize(
    ("backend", "impl"),
    [("brax", None), ("playground", "jax"), ("playground", "warp")],
    ids=["brax", "playground_jax", "playground_warp"],
)
def test_native_metric_reward_selection_and_factory_override(backend, impl):
    if backend == "playground":
        if impl == "warp" and jax.default_backend() != "gpu":
            pytest.skip("Playground's Warp backend requires a GPU")
        env_name = "CartpoleBalance"
        reward_keys = ("reward/upright", "reward/centered")
        kwargs = {"config_overrides": {"impl": impl}}
    else:
        env_name = "reacher"
        reward_keys = ("reward_dist", "reward_ctrl")
        kwargs = {}
    cfg = OmegaConf.create(
        {"env_type": backend, "env_name": env_name, "reward_keys": reward_keys}
    )
    env = create_env(
        cfg,
        parallel=2,
        episode_length=1,
        autoreset_mode=AutoresetMode.DISABLED,
        **kwargs,
    )
    wrapper_cls = (
        BraxMultiObjectiveRewardWrapper
        if backend == "brax"
        else MujocoMultiObjectiveRewardWrapper
    )
    assert type(get_wrapper(env, wrapper_cls)) is wrapper_cls
    keys = jax.random.split(jax.random.key(5), (2, 2))

    def run(key):
        state = env.reset(key)
        next_state = env.step(state, jnp.zeros((2, *env.action_space.shape)))
        return state, next_state

    reset, state = jax.jit(jax.vmap(jax.vmap(run)))(keys)
    assert reset.reward.shape == (2, 2, 2, 2)
    assert state.done.shape == (2, 2, 2)
    np.testing.assert_array_equal(reset.reward, 0)
    np.testing.assert_allclose(
        state.reward,
        jnp.stack([state.info.metrics[name] for name in reward_keys], axis=-1),
    )
    assert state.env_state.reward.shape == (2, 2, 2)

    # An explicit None must override configured reward keys and retain scalars.
    scalar_env = create_env(cfg, reward_keys=None, parallel=2, **kwargs)
    assert jax.eval_shape(scalar_env.reset, jax.random.key(0)).reward.shape == (2,)
    one_env = create_env(cfg, reward_keys=reward_keys[:1], parallel=2, **kwargs)
    assert jax.eval_shape(one_env.reset, jax.random.key(0)).reward.shape == (2, 1)
