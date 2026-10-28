"""PCN complete episodic learning, conditioned evaluation and state isolation."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from flax import nnx
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.agent import AgentState
from evorl.algorithms.multi_objective.pcn import PCNWorkflow, sample_episode_batch
from evorl.envs import Env, EnvState, Box
from evorl.envs.wrappers import OneEpisodeWrapper, VmapWrapper
from evorl.types import PyTreeDict
from evorl.utils.orbax_utils import load, save
from tests.helpers.multi_objective_envs import ToyMultiObjectiveEnv

pytestmark = pytest.mark.integration


def make_workflow(
    *,
    jit=True,
    continuous=False,
    objectives=2,
    native=False,
    overrides=None,
    multi_device=False,
    random_resets=False,
):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "configs"),
        version_base=None,
    ):
        config = compose(
            config_name="config",
            overrides=["agent=exp/multi_objective/pcn/deep-sea-treasure"],
        )
    config = OmegaConf.merge(
        config,
        dict(
            num_envs=2,
            num_eval_envs=2,
            num_initial_episodes=4,
            episode_buffer_capacity=6,
            num_command_candidates=4,
            num_eval_commands=2,
            num_updates_per_iter=2,
            batch_size=4,
            total_timesteps=16,
            eval_interval=1,
            env=dict(max_episode_steps=4),
            agent_network=dict(hidden_size=4),
            scaling_factor=None,
            max_return=None,
        ),
        overrides or {},
    )
    cls = type("TestPCNWorkflow", (PCNWorkflow,), {})
    if native:
        return cls.build_from_config(config, jit=jit, multi_device=multi_device)

    def create_env(config, parallel, **kwargs):
        class RandomResetEnv(ToyMultiObjectiveEnv):
            def reset(self, key):
                state = super().reset(key)
                return state.replace(obs=state.obs.at[-1].set(jax.random.uniform(key)))

        env_cls = RandomResetEnv if random_resets else ToyMultiObjectiveEnv
        return VmapWrapper(
            OneEpisodeWrapper(
                env_cls(continuous, objectives), config.max_episode_steps
            ),
            parallel,
            vmap_step=True,
        )

    with patch(
        "evorl.algorithms.multi_objective.pcn.create_env", side_effect=create_env
    ):
        return cls.build_from_config(config, jit=jit, multi_device=multi_device)


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("objectives", [1, 2, 3])
def test_pcn_training_and_command_evaluation_are_functional(
    continuous, objectives, jit
):
    workflow = make_workflow(jit=jit, continuous=continuous, objectives=objectives)
    try:
        state = workflow.setup(jax.random.key(1))
        assert state.agent_state.obs_preprocessor_state is None
        before = jax.tree.map(
            np.array, (state.agent_state, state.opt_state, state.episode_archive)
        )
        metrics, updated = workflow.step(state)
        assert np.isfinite(metrics.loss)
        assert int(state.metrics.sampled_timesteps) == 8
        assert int(updated.metrics.sampled_timesteps) == 12
        assert int(state.metrics.sampled_episodes) == 0
        assert int(updated.metrics.sampled_episodes) == 2
        assert int(updated.metrics.training_updates) == 2
        archive = updated.episode_archive
        assert archive.returns.shape == (6, 4, objectives)
        np.testing.assert_array_equal(archive.lengths, 2)
        np.testing.assert_array_equal(archive.valid[:, 2:], False)
        np.testing.assert_array_equal(archive.rewards[:, 2:], 0)
        # Final observation stays at the terminal count, never at reset count.
        np.testing.assert_array_equal(archive.next_obs[:, 1, 0], 2)
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(
                jax.tree.leaves(state.agent_state.model_state),
                jax.tree.leaves(updated.agent_state.model_state),
                strict=True,
            )
        )
        evaluation, evaluated = workflow.evaluate(updated)
        assert evaluation.episode_return.shape == (2, objectives)
        assert evaluation.desired_return.shape == (2, objectives)
        np.testing.assert_allclose(evaluation.episode_length, 2)
        np.testing.assert_allclose(
            evaluation.command_error,
            np.linalg.norm(
                evaluation.episode_return - evaluation.desired_return, axis=-1
            ),
        )
        assert evaluated.metrics.sampled_timesteps == updated.metrics.sampled_timesteps
        for previous, current in zip(
            jax.tree.leaves(before),
            jax.tree.leaves(
                (state.agent_state, state.opt_state, state.episode_archive)
            ),
            strict=True,
        ):
            np.testing.assert_array_equal(previous, current)
    finally:
        workflow.close()


@pytest.mark.parametrize("multi_device", [False, True])
@pytest.mark.parametrize("normalize_obs", [False, True])
def test_pcn_nested_vmap_matches_independent_instances(
    multi_device, normalize_obs, dp_mesh
):
    workflow = make_workflow(
        jit=False,
        multi_device=multi_device,
        random_resets=normalize_obs,
        overrides=dict(normalize_obs=normalize_obs),
    )
    try:
        keys = jax.random.split(jax.random.key(2), (2, 2))
        states = jax.jit(jax.vmap(jax.vmap(workflow.setup)))(keys)
        metrics, updated = jax.jit(jax.vmap(jax.vmap(workflow.step)))(states)
        evaluation, _ = jax.jit(jax.vmap(jax.vmap(workflow.evaluate)))(updated)
        assert metrics.loss.shape == (2, 2)
        assert evaluation.episode_return.shape == (2, 2, 2, 2)
        independent = workflow.setup(keys[1, 0])
        _, independent = workflow.step(independent)
        for batched, separate in zip(
            jax.tree.leaves(updated), jax.tree.leaves(independent), strict=True
        ):
            if jnp.issubdtype(batched.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(
                    jax.random.key_data(batched[1, 0]), jax.random.key_data(separate)
                )
            else:
                np.testing.assert_allclose(
                    batched[1, 0], separate, rtol=3e-5, atol=2e-6
                )
    finally:
        workflow.close()


@pytest.mark.parametrize("multi_device", [False, True])
def test_observation_statistics_count_real_collections_once(multi_device, jit):
    """Count prefill and later observations, excluding padding and replay reuse."""
    workflow = make_workflow(
        jit=jit,
        multi_device=multi_device,
        overrides=dict(normalize_obs=True, episode_buffer_capacity=4),
    )
    try:
        with jax.set_mesh(workflow.mesh) if multi_device else nullcontext():
            state = workflow.setup(jax.random.key(71))
            for iteration, count in enumerate((8, 12, 16)):
                if iteration:
                    _, state = workflow.step(state)
                stats = state.agent_state.obs_preprocessor_state
                assert int(stats.count) == count
                assert int(state.metrics.sampled_timesteps) == count
                np.testing.assert_allclose(stats.mean, [0.5, 1, -1])
                np.testing.assert_allclose(stats.std, [0.5, 1e-6, 1e-6], rtol=1e-6)
                # Four-step records contain two real transitions. Eviction does not
                # remove observations from the cumulative statistics.
                np.testing.assert_array_equal(state.episode_archive.valid[:, 2:], False)
                before = jax.tree.map(np.array, stats)
                _, evaluated = workflow.evaluate(state)
                for a, b in zip(
                    jax.tree.leaves(before),
                    jax.tree.leaves(evaluated.agent_state.obs_preprocessor_state),
                    strict=True,
                ):
                    np.testing.assert_array_equal(a, b)
    finally:
        workflow.close()


def test_real_deep_sea_treasure_terminates_once_without_padding_rewards():
    workflow = make_workflow(native=True)
    try:
        state = workflow.setup(jax.random.key(3))
        models = workflow.agent.get_models(state.agent_state, copy=True)
        nnx.update(models, jax.tree.map(jnp.zeros_like, nnx.state(models, nnx.Param)))
        models.policy.output.bias[...] = jnp.array([-100.0, 100.0, -100.0, -100.0])
        agent_state = state.agent_state.replace(model_state=nnx.unpack(models))
        episodes = jax.jit(
            lambda key: workflow._collect_episodes(
                workflow.env,
                agent_state,
                key,
                jnp.array([0.7, -1]),
                jnp.float32(1),
                evaluate=True,
            )
        )(jax.random.key(4))
        np.testing.assert_array_equal(episodes.lengths, 1)
        np.testing.assert_array_equal(episodes.valid[:, 1:], False)
        np.testing.assert_allclose(episodes.returns[:, 0], [[0.7, -1], [0.7, -1]])
        np.testing.assert_array_equal(episodes.rewards[:, 1:], 0)
        np.testing.assert_array_equal(episodes.next_obs[:, 0], [[1, 0], [1, 0]])
        metrics, updated = workflow.step(state)
        assert np.isfinite(metrics.loss)
        assert int(updated.metrics.sampled_timesteps) > int(
            state.metrics.sampled_timesteps
        )
    finally:
        workflow.close()


@pytest.mark.parametrize("normalize_obs", [False, True])
def test_real_deep_sea_treasure_data_parallel_training(normalize_obs):
    workflow = make_workflow(
        native=True,
        multi_device=True,
        overrides=dict(normalize_obs=normalize_obs),
    )
    try:
        with jax.set_mesh(workflow.mesh):
            state = workflow.setup(jax.random.key(43))
            metric, updated = workflow.step(state)
            evaluation, _ = workflow.evaluate(updated)
            assert np.isfinite(metric.loss)
            assert np.isfinite(evaluation.episode_return).all()
            assert int(state.metrics.sampled_episodes) == 0
            assert int(updated.metrics.sampled_episodes) == 2
            assert int(updated.metrics.sampled_timesteps) > int(
                state.metrics.sampled_timesteps
            )
    finally:
        workflow.close()


@pytest.mark.parametrize("nested", [False, True], ids=["single", "nested"])
@pytest.mark.parametrize("normalize_obs", [False, True])
def test_real_mo_gymnasium_complete_pcn_training(
    mo_gymnasium_pools, nested, normalize_obs
):
    workflow = make_workflow(
        native=True,
        jit=not nested,
        overrides=dict(env=dict(env_type="mo_gymnasium"), normalize_obs=normalize_obs),
    )
    try:

        def run(key):
            state = workflow.setup(key)
            before = state.agent_state.model_state
            training, state = workflow.step(state)
            evaluation, _ = workflow.evaluate(state)
            return before, state, training, evaluation

        if nested:
            keys = jax.random.split(jax.random.key(12), (2, 2))
            before, state, training, evaluation = jax.jit(jax.vmap(jax.vmap(run)))(keys)
            batch_shape = (2, 2)
        else:
            before, state, training, evaluation = run(jax.random.key(12))
            batch_shape = ()
        jax.block_until_ready(state)
        assert state.episode_archive.returns.shape == (*batch_shape, 6, 4, 2)
        assert evaluation.episode_return.shape == (*batch_shape, 2, 2)
        assert np.isfinite(training.loss).all()
        assert np.isfinite(evaluation.episode_return).all()
        assert np.all(np.asarray(evaluation.episode_length) > 0)
        assert np.all(np.asarray(evaluation.episode_length) <= 4)
        np.testing.assert_array_equal(state.metrics.sampled_episodes, 2)
        np.testing.assert_array_equal(state.metrics.training_updates, 2)
        # The archive contains four prefill episodes and two training episodes.
        # The timestep counter includes both, while the episode counter excludes prefill.
        np.testing.assert_array_equal(
            state.metrics.sampled_timesteps,
            np.asarray(state.episode_archive.lengths).sum(-1),
        )
        assert any(
            not np.array_equal(old, new)
            for old, new in zip(
                jax.tree.leaves(before),
                jax.tree.leaves(state.agent_state.model_state),
                strict=True,
            )
        )
    finally:
        workflow.close()


class CommandEchoEnv(Env):
    """Expose the selected command-action in the next observation."""

    @property
    def obs_space(self):
        return Box(low=-jnp.ones(2) * 100, high=jnp.ones(2) * 100)

    @property
    def action_space(self):
        return self.obs_space

    def reset(self, key):
        return EnvState(
            env_state=jnp.int32(0),
            obs=jnp.zeros(2),
            reward=jnp.zeros(1),
            done=jnp.float32(0),
        )

    def step(self, state, action):
        return state.replace(
            env_state=state.env_state + 1,
            obs=action,
            reward=jnp.ones(1),
            done=(state.env_state >= 2).astype(jnp.float32),
        )


class CommandEchoAgent:
    """Return explicit commands, making their update observable in a test."""

    num_objectives = 1

    def evaluate_actions(self, state, batch, key):
        return jnp.stack(
            (batch.extras.desired_return[:, 0], batch.extras.desired_horizon), axis=-1
        ), PyTreeDict()


def test_discounted_return_command_is_rescaled_and_horizon_stays_positive():
    workflow = make_workflow(overrides=dict(discount=0.5))
    try:
        workflow.agent = CommandEchoAgent()
        env = VmapWrapper(OneEpisodeWrapper(CommandEchoEnv(), 4), 2, vmap_step=True)
        episodes = workflow._collect_episodes(
            env,
            AgentState(),
            jax.random.key(5),
            jnp.array([5.0]),
            jnp.float32(2),
            evaluate=True,
        )
        np.testing.assert_allclose(episodes.actions[0, :3], [[5, 2], [8, 1], [14, 1]])
        np.testing.assert_allclose(episodes.returns[0, :3, 0], [1.75, 1.5, 1])
        np.testing.assert_array_equal(episodes.lengths, 3)
    finally:
        workflow.close()


@pytest.mark.parametrize("multi_device", [False, True])
def test_learn_budget_records_and_checkpoint_round_trip(
    tmp_path, multi_device, dp_mesh
):
    workflow = make_workflow(
        overrides=dict(total_timesteps=13), multi_device=multi_device
    )
    try:
        workflow.recorder = Mock()
        workflow.checkpoint_manager = Mock()
        state = workflow.setup(jax.random.key(6))
        updated = workflow.learn(state)
        assert int(updated.metrics.sampled_timesteps) == 16
        assert int(updated.metrics.iterations) == 2
        assert workflow.recorder.write.call_count == 6
        workflow.checkpoint_manager.save.assert_called_with(2, updated, force=True)
        path = str(tmp_path / "checkpoint")
        save(path, updated)
        restored = load(path, updated)
        metric_a, next_a = workflow.step(updated)
        metric_b, next_b = workflow.step(restored)
        np.testing.assert_allclose(metric_a.loss, metric_b.loss)
        for a, b in zip(jax.tree.leaves(next_a), jax.tree.leaves(next_b), strict=True):
            if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(
                    jax.random.key_data(a), jax.random.key_data(b)
                )
            else:
                np.testing.assert_allclose(a, b)
    finally:
        workflow.close()


def test_learn_saves_final_real_checkpoint_without_duplicate_steps(tmp_path):
    workflow = make_workflow(
        overrides=dict(
            total_timesteps=13,
            checkpoint=dict(enable=True, save_interval_steps=100),
            output_dir=str(tmp_path),
        )
    )
    try:
        state = workflow.learn(workflow.setup(jax.random.key(8)))
        workflow.checkpoint_manager.wait_until_finished()
        assert workflow.checkpoint_manager.latest_step() == 2
        restored = workflow.checkpoint_manager.restore(2, state)
        np.testing.assert_array_equal(
            restored.episode_archive.returns, state.episode_archive.returns
        )
        metric_a, next_a = workflow.step(state)
        metric_b, next_b = workflow.step(restored)
        np.testing.assert_allclose(metric_a.loss, metric_b.loss)
        np.testing.assert_array_equal(
            next_a.episode_archive.returns, next_b.episode_archive.returns
        )
    finally:
        workflow.close()


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(num_initial_episodes=3), "divisible"),
        (dict(episode_buffer_capacity=1), "episode_buffer_capacity"),
        (dict(num_command_candidates=10), "Command counts"),
        (dict(scaling_factor=[1, 2]), "scaling_factor"),
        (dict(discount=0), "discount"),
        (dict(max_return=[1]), "max_return"),
    ],
)
def test_pcn_rejects_invalid_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_workflow(overrides=overrides)


def test_pcn_two_device_gradient_matches_combined_sample_batch(dp_mesh):
    workflow = make_workflow(multi_device=True, overrides=dict(num_updates_per_iter=1))
    try:
        with jax.set_mesh(workflow.mesh):
            state = workflow.setup(jax.random.key(41))
            _, sample_key = jax.random.split(state.key)
            batches = jax.vmap(
                lambda rank: sample_episode_batch(
                    state.episode_archive,
                    jax.random.fold_in(sample_key, rank),
                    workflow.config.batch_size,
                )
            )(jnp.arange(2))
            batch = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), batches)
            models = workflow.agent.get_models(state.agent_state, copy=True)

            def loss_fn(models):
                return workflow.agent.loss(
                    state.agent_state.replace(model_state=nnx.unpack(models)),
                    batch,
                    sample_key,
                ).policy_loss

            expected_loss, grads = nnx.value_and_grad(loss_fn)(models)
            optimizer = nnx.merge(state.opt_state, copy=True)
            optimizer.update(models.policy, grads.policy)
            metric, updated = workflow.step(state)
            np.testing.assert_allclose(metric.loss, expected_loss, rtol=1e-6)
            for actual, expected in zip(
                jax.tree.leaves(updated.opt_state),
                jax.tree.leaves(nnx.unpack(optimizer)),
                strict=True,
            ):
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    finally:
        workflow.close()


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("normalize_obs", [False, True])
def test_pcn_data_parallel_global_archive_and_replicated_updates(
    continuous, normalize_obs
):
    workflow = make_workflow(
        continuous=continuous,
        multi_device=True,
        random_resets=True,
        overrides=dict(normalize_obs=normalize_obs),
    )
    try:
        with jax.set_mesh(workflow.mesh):
            state = workflow.setup(jax.random.key(29))
            assert int(state.metrics.sampled_episodes) == 0
            assert int(state.metrics.sampled_timesteps) == 8
            assert state.episode_archive.returns.shape == (6, 4, 2)
            metric, updated = workflow.step(state)
            assert np.isfinite(metric.loss)
            assert int(updated.metrics.sampled_episodes) == 2
            assert int(updated.metrics.sampled_timesteps) == 12
            assert int(updated.metrics.training_updates) == 2
            if normalize_obs:
                for current in (state, updated):
                    stats = current.agent_state.obs_preprocessor_state
                    observations = np.asarray(current.episode_archive.obs)[
                        np.asarray(current.episode_archive.valid)
                    ]
                    assert int(stats.count) == len(observations)
                    np.testing.assert_allclose(
                        stats.mean, observations.mean(0), rtol=1e-5, atol=1e-6
                    )
                    np.testing.assert_allclose(
                        stats.std,
                        np.maximum(observations.std(0), 1e-6),
                        rtol=1e-5,
                        atol=1e-6,
                    )
            for leaf in jax.tree.leaves(updated):
                assert leaf.sharding.is_fully_replicated
                shards = leaf.addressable_shards
                if not jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
                    for shard in shards[1:]:
                        np.testing.assert_array_equal(shards[0].data, shard.data)
            # Each local batch contributes distinct reset observations to the
            # one global archive, rather than retaining separate local fronts.
            archive = np.asarray(updated.episode_archive.obs)
            assert len(np.unique(archive[:, 0], axis=0)) >= 4
            evaluation, _ = workflow.evaluate(updated)
            np.testing.assert_allclose(
                evaluation.command_error,
                np.linalg.norm(
                    evaluation.episode_return - evaluation.desired_return, axis=-1
                ),
            )
    finally:
        workflow.close()
