"""Functional off-policy workflow updates, schedules and NNX state ownership."""

from pathlib import Path
from unittest.mock import patch

from flax import nnx
from hydra.utils import get_class
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.algorithms.td3 import TD3Agent
from evorl.networks.linear import QNetwork
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from tests.helpers.envs import ToyEnv


class CountingQNetwork(QNetwork):
    """A real critic with non-parameter state updated by every forward call."""

    def __init__(self, critic, calls):
        super().__init__(critic.mlp, critic.n_stack, critic.obs_key)
        self.calls = nnx.Variable(jnp.int32(calls))

    def __call__(self, obs, actions):
        self.calls[...] += 1
        return super().__call__(obs, actions)


def assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)


def make_workflow(name, jit, multi_device=False, **overrides):
    root = Path(__file__).resolve().parents[2]
    agent_name = {"sac_discrete": "sac", "td3_v2": "contrib/td3-v2"}.get(name, name)
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / f"configs/agent/{agent_name}.yaml"),
        dict(
            env=dict(max_episode_steps=2),
            num_envs=4,
            num_eval_envs=4,
            eval_episodes=4,
            normalize_obs=True,
            rollout_length=1,
            batch_size=4,
            replay_buffer_capacity=32,
            random_timesteps=4,
            learning_start_timesteps=4,
            num_updates_per_iter=2,
            tau=1.0,
            agent_network=dict(
                actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4]
            ),
        ),
    )
    if agent_name in ("td3", "sac"):
        config.actor_update_interval = 3
    if agent_name == "sac":
        config.adaptive_alpha = True
    if agent_name == "dqn":
        config.agent_network.q_hidden_layer_sizes = [4]
        config.target_network_update_freq = 2
    if name == "td7":
        config.rollout_episodes = 4
        config.target_update_rate = 1
        config.policy_freq = 1
        config.agent_network = dict(
            zs_dim=4,
            zsa_dim=4,
            f_layer_sizes=[4],
            g_layer_sizes=[4],
            actor_hidden_layer_sizes=[4],
            critic_hidden_layer_sizes=[4],
            state_emb_dim=4,
            state_action_emb_dim=4,
        )
    config = OmegaConf.merge(config, overrides)
    base = get_class(config.workflow_cls)
    cls = type(f"NNX{name}Workflow", (base,), {})

    def create_env(*args, **kwargs):
        return ToyEnv(kwargs["parallel"], discrete=name in ("sac_discrete", "dqn"))

    with patch(
        f"{base._build_from_config.__module__}.create_env", side_effect=create_env
    ):
        workflow = cls.build_from_config(config, multi_device=multi_device, jit=jit)
    return workflow


@pytest.mark.integration
@pytest.mark.parametrize("name", ["td3", "sac", "sac_discrete", "ddpg", "dqn"])
def test_workflow_update_counts_and_target_copies_follow_algorithm_schedule(name, jit):
    workflow = make_workflow(name, jit)
    agent_name = "sac" if name == "sac_discrete" else name
    try:
        state = workflow.setup(jax.random.key(0))
        metrics, next_state = workflow.step(state)
        assert int(next_state.metrics.iterations) == 1
        assert np.isfinite(jax.tree.leaves(metrics)[0]).all()
        if name == "dqn":
            assert int(nnx.merge(next_state.opt_state, copy=True).step[...]) == 2
            assert int(next_state.metrics.training_updates) == 2
        else:
            assert int(nnx.merge(next_state.opt_state.actor, copy=True).step[...]) == 2
            expected_critic_steps = 6 if agent_name in ("td3", "sac") else 2
            assert (
                int(nnx.merge(next_state.opt_state.critic, copy=True).step[...])
                == expected_critic_steps
            )
            if agent_name == "sac":
                assert (
                    int(nnx.merge(next_state.opt_state.alpha, copy=True).step[...]) == 2
                )
        params = nnx.state(
            nnx.merge(next_state.agent_state.model_state, copy=True), nnx.Param
        )
        if name == "dqn":
            assert_tree_equal(params["target_q"], params["q"])
        else:
            assert_tree_equal(params["target_critic"], params["critic"])
            if name in ("td3", "ddpg"):
                assert_tree_equal(params["target_actor"], params["actor"])
        for x in jax.tree.leaves(next_state.agent_state.model_state):
            assert np.isfinite(x).all()
    finally:
        workflow.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    "name", ["td3", "td7", "ddpg", "sac", "sac_discrete", "dqn", "td3_v2"]
)
def test_step_matches_explicit_parameter_gradients(name, jit):
    """Complete training agrees with pure JAX differentiation of selected params.

    Exercise the public step rather than expose its nested update functions for
    tests. SAC and TD3 also exercise three-critic ensembles.
    """
    overrides = dict(num_updates_per_iter=1)
    if name == "td3_v2":
        overrides["actor_update_interval"] = 1
    if name in ("td3", "sac", "sac_discrete", "td3_v2"):
        overrides["agent_network"] = dict(num_critics=3)
    workflow = make_workflow(name, jit=False, **overrides)

    def explicit_loss_and_grad(loss_fn, axis_name, has_aux=False, *, wrt):
        assert axis_name is None
        assert has_aux

        def update(models, *args, **kwargs):
            model_state = nnx.unpack(models)
            params = nnx.state(models, wrt)

            def objective(params):
                local_models = nnx.merge(model_state, copy=True)
                nnx.update(local_models, params)
                loss, aux = loss_fn(local_models, *args, **kwargs)
                return loss, (aux, nnx.unpack(local_models))

            (loss, (aux, next_model_state)), grads = jax.value_and_grad(
                objective, has_aux=True
            )(params)
            # Loss evaluation can update auxiliary variables, e.g. spectral norm.
            nnx.update(models, nnx.state(nnx.merge(next_model_state, copy=True)))
            return (loss, aux), grads

        return update

    try:
        state = workflow.setup(jax.random.key(30))
        original = jax.tree.map(lambda x: x.copy(), state)
        step = jax.jit(workflow.step) if jit else workflow.step
        actual = step(state)
        with patch(
            f"{workflow.step.__module__}.pmean_nnx_loss_and_grad",
            explicit_loss_and_grad,
        ):
            # Use a separate compilation so the patched gradient is traced.
            reference_step = lambda state: workflow.step(state)
            expected = (jax.jit(reference_step) if jit else reference_step)(state)
        for got, want in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            if jax.dtypes.issubdtype(got.dtype, jax.dtypes.prng_key):
                got, want = jax.random.key_data(got), jax.random.key_data(want)
            np.testing.assert_allclose(got, want, rtol=3e-5, atol=2e-6)
        for got, want in zip(
            jax.tree.leaves(state), jax.tree.leaves(original), strict=True
        ):
            if jax.dtypes.issubdtype(got.dtype, jax.dtypes.prng_key):
                got, want = jax.random.key_data(got), jax.random.key_data(want)
            np.testing.assert_array_equal(got, want)
    finally:
        workflow.close()


@pytest.mark.integration
def test_td3_actor_update_preserves_critic_auxiliary_state(jit):
    """Actor-only forward passes must not persist critic spectral estimates."""
    workflow = make_workflow(
        "td3", jit=False, agent_network=dict(norm_layer_type="spectral_norm")
    )
    actor_loss = TD3Agent.actor_loss

    def actor_loss_without_critic_mutation(agent, agent_state, *args, **kwargs):
        models = agent.get_models(agent_state)
        critic_state = nnx.clone(nnx.state(models.critic, nnx.Not(nnx.Param)))
        result = actor_loss(agent, agent_state, *args, **kwargs)
        nnx.update(models.critic, critic_state)
        return result

    try:
        state = workflow.setup(jax.random.key(5))
        step = jax.jit(workflow.step) if jit else workflow.step
        _, actual = step(state)
        with patch.object(TD3Agent, "actor_loss", actor_loss_without_critic_mutation):
            reference_step = lambda state: workflow.step(state)
            _, expected = (jax.jit(reference_step) if jit else reference_step)(state)
        for got, want in zip(
            jax.tree.leaves((actual.agent_state, actual.opt_state)),
            jax.tree.leaves((expected.agent_state, expected.opt_state)),
            strict=True,
        ):
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        before = nnx.merge(state.agent_state.model_state, copy=True)
        after = nnx.merge(actual.agent_state.model_state, copy=True)
        for component in ("actor", "critic", "target_actor", "target_critic"):
            initial_aux = jax.tree.leaves(
                nnx.state(before[component], nnx.Not(nnx.Param))
            )
            updated_aux = jax.tree.leaves(
                nnx.state(after[component], nnx.Not(nnx.Param))
            )
            assert initial_aux
            assert any(
                not np.array_equal(a, b)
                for a, b in zip(initial_aux, updated_aux, strict=True)
            )
    finally:
        workflow.close()


@pytest.mark.integration
def test_td3_spectral_step_matches_full_batch_on_two_devices(dp_mesh, jit):
    """Whole-step loss, gradients and auxiliary state match full-batch training."""
    obs = np.arange(12, dtype=np.float32).reshape(4, 3) / 10
    batch = SampleBatch(
        obs=obs,
        actions=np.full((4, 1), 0.2, dtype=np.float32),
        rewards=np.array([1.0, -0.3, 0.5, 0.1], dtype=np.float32),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=obs + 0.1,
                termination=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            )
        ),
    )
    config = dict(
        normalize_obs=False,
        policy_noise=0.0,
        actor_update_interval=1,
        num_updates_per_iter=1,
        agent_network=dict(norm_layer_type="spectral_norm"),
    )
    local = make_workflow("td3", jit, **config)
    distributed = make_workflow("td3", jit, multi_device=True, **config)

    def fixed_batch(buffer, state, key):
        count = buffer.sample_batch_size
        index = jax.lax.axis_index("DP") * count if count < 4 else 0
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                jnp.asarray(x), index, count, axis=0
            ),
            batch,
        )

    try:
        local_state = local.setup(jax.random.key(17))
        with jax.set_mesh(distributed.mesh):
            distributed_state = distributed.setup(jax.random.key(17))
            with patch("evorl.replay_buffers.ReplayBuffer.sample", fixed_batch):
                expected_metric, expected_state = local.step(local_state)
                actual_metric, actual_state = distributed.step(distributed_state)
        for got, want in zip(
            jax.tree.leaves(
                (actual_metric, actual_state.agent_state, actual_state.opt_state)
            ),
            jax.tree.leaves(
                (expected_metric, expected_state.agent_state, expected_state.opt_state)
            ),
            strict=True,
        ):
            np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)
    finally:
        local.close()
        distributed.close()


@pytest.mark.integration
def test_td3_critic_forward_variable_updates_survive_state_loss_adapter(jit):
    """State-based losses retain forward mutations through NNX differentiation.

    Count actual critic forwards, rather than infer mutation from spectral norm:
    each critic update calls online and target critics once. Actor-only forwards
    are deliberately excluded from the online critic's persistent state.
    """
    workflow = make_workflow(
        "td3",
        jit=False,
        normalize_obs=False,
        actor_update_interval=3,
        num_updates_per_iter=2,
    )
    obs = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
    batch = SampleBatch(
        obs=obs,
        actions=jnp.full((4, 1), 0.2),
        rewards=jnp.ones(4),
        extras=PyTreeDict(
            env_extras=PyTreeDict(ori_obs=obs + 0.1, termination=jnp.zeros(4))
        ),
    )
    try:
        state = workflow.setup(jax.random.key(41))
        models = workflow.agent.get_models(state.agent_state, copy=True)
        models.critic = CountingQNetwork(models.critic, calls=7)
        models.target_critic = CountingQNetwork(models.target_critic, calls=19)
        state = state.replace(
            agent_state=state.agent_state.replace(model_state=nnx.unpack(models))
        )
        original_agent = jax.tree.map(np.array, state.agent_state)
        original_optimizer = jax.tree.map(np.array, state.opt_state)
        initial_params = nnx.state(models.critic, nnx.Param)
        assert "calls" not in initial_params

        step = jax.jit(workflow.step) if jit else workflow.step
        with patch("evorl.replay_buffers.ReplayBuffer.sample", return_value=batch):
            _, first = step(state)
            _, repeated = step(state)
            _, second = step(first)

        for iteration, result in enumerate((first, second), start=1):
            updated = workflow.agent.get_models(result.agent_state, copy=True)
            # 2 iterations of 3 critic updates; the 2 actor forwards are restored.
            np.testing.assert_array_equal(updated.critic.calls[...], 7 + 6 * iteration)
            np.testing.assert_array_equal(
                updated.target_critic.calls[...], 19 + 6 * iteration
            )
            assert int(nnx.merge(result.opt_state.critic, copy=True).step[...]) == (
                6 * iteration
            )
            assert int(nnx.merge(result.opt_state.actor, copy=True).step[...]) == (
                2 * iteration
            )
            assert any(
                not np.array_equal(before, after)
                for before, after in zip(
                    jax.tree.leaves(initial_params),
                    jax.tree.leaves(nnx.state(updated.critic, nnx.Param)),
                    strict=True,
                )
            )
        assert_tree_equal(first.agent_state, repeated.agent_state)
        assert_tree_equal(state.agent_state, original_agent)
        assert_tree_equal(state.opt_state, original_optimizer)
    finally:
        workflow.close()
