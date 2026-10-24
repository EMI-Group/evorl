"""MORL Bellman operators and losses checked against independent calculations."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.algorithms.multi_objective.capql import CAPQLAgent
from evorl.algorithms.multi_objective.common import sample_preferences
from evorl.algorithms.multi_objective.envelope_q import (
    EnvelopeQAgent,
    envelope_next_values,
)
from evorl.distribution import get_tanh_norm_dist
from evorl.envs import Box, Discrete
from evorl.envs.wrappers import ActionSquashWrapper
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from tests.helpers.multi_objective_envs import ToyMultiObjectiveEnv


@pytest.mark.parametrize("evaluate", [False, True])
def test_action_bounds_are_handled_only_by_the_environment(evaluate):
    env = ActionSquashWrapper(ToyMultiObjectiveEnv(True))
    kwargs = dict(
        num_objectives=2,
        agent_network_cfg=OmegaConf.create(
            dict(
                num_critics=2, actor_hidden_layer_sizes=[], critic_hidden_layer_sizes=[]
            )
        ),
    )
    agent = CAPQLAgent(**kwargs)
    state = agent.init(env.obs_space, env.action_space, jax.random.key(33))
    assert set(state.extra_state) == {"preferences"}
    action_fn = agent.evaluate_actions if evaluate else agent.compute_actions
    env_state = env.reset(jax.random.key(34))
    action, _ = jax.jit(action_fn)(
        state, SampleBatch(obs=env_state.obs), jax.random.key(35)
    )
    assert bool(jnp.all((action >= -1) & (action <= 1)))
    physical = action * jnp.array([2.0, 1.5]) + jnp.array([0.0, 1.5])
    expected = env.env.step(env_state, physical)
    actual = jax.jit(env.step)(env_state, action)
    np.testing.assert_allclose(actual.reward, expected.reward, atol=1e-6)


def test_envelope_operator_selects_one_vector_across_preferences_and_actions(jit):
    online = np.array(
        [
            [[[9, 1], [4, 3]], [[5, 7], [2, 10]], [[8, 6], [0, 0]]],
            [[[1, 9], [7, 3]], [[3, 5], [8, 2]], [[6, 6], [4, 8]]],
        ],
        np.float32,
    )
    target = -online + np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    weights = np.array([[1, 0], [0.2, 0.8], [0.6, 0.4]], np.float32)
    expected = np.empty((2, 3, 2), np.float32)
    for batch in range(2):
        for query, weight in enumerate(weights):
            # Selection uses online scalar utility; gather the whole target
            # vector from the same candidate/action pair.
            index = np.argmax((online[batch] * weight).sum(-1))
            candidate, action = np.unravel_index(index, (3, 2))
            expected[batch, query] = target[batch, candidate, action]
    fn = jax.jit(envelope_next_values) if jit else envelope_next_values
    actual = fn(*map(jnp.asarray, (online, target, weights)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("num_objectives", [1, 2, 4])
def test_preference_samples_have_unit_mass_under_nested_vmap(num_objectives):
    keys = jax.random.split(jax.random.key(3), (2, 3))
    values = jax.jit(
        jax.vmap(jax.vmap(lambda key: sample_preferences(key, (7,), num_objectives)))
    )(keys)
    assert values.shape == (2, 3, 7, num_objectives)
    assert np.all(values >= 0)
    np.testing.assert_allclose(values.sum(-1), 1, atol=2e-7)


def envelope_q_agent():
    agent = EnvelopeQAgent(
        num_objectives=2,
        agent_network_cfg=OmegaConf.create(dict(q_hidden_layer_sizes=[])),
        discount=0.9,
        num_sample_preferences=3,
    )
    obs = Box(low=-jnp.ones(3), high=jnp.ones(3))
    state = agent.init(obs, Discrete(n=2), jax.random.key(0))
    models = agent.get_models(state, copy=True)
    nnx.update(models, jax.tree.map(jnp.zeros_like, nnx.state(models, nnx.Param)))
    models.q.mlp.layers[0].bias[...] = jnp.array([5.0, 1.0, 2.0, 6.0])
    models.target_q.mlp.layers[0].bias[...] = jnp.array([3.0, 2.0, 1.0, 4.0])
    return agent, state.replace(model_state=nnx.unpack(models))


@pytest.mark.parametrize("mix", [0.0, 0.35, 1.0])
def test_envelope_homotopy_loss_and_bootstrap_match_numpy(mix, jit):
    agent, state = envelope_q_agent()
    state = state.replace(
        extra_state=state.extra_state.replace(homotopy_lambda=jnp.float32(mix))
    )
    batch = SampleBatch(
        obs=jnp.ones((3, 3)),
        actions=jnp.array([0, 1, 0]),
        rewards=jnp.array([[1.0, 2.0], [-1.0, 3.0], [2.0, -2.0]]),
        dones=jnp.array([1.0, 1.0, 0.0]),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=jnp.ones((3, 3)) * 2,
                termination=jnp.array([1.0, 0.0, 0.0]),
            ),
            preferences=jnp.ones((3, 2)) / 2,
        ),
    )
    key = jax.random.key(5)
    weights = np.abs(np.array(jax.random.normal(key, (3, 2))))
    weights /= weights.sum(-1, keepdims=True)
    online = np.array([[5, 1], [2, 6]], np.float32)
    target = np.array([[3, 2], [1, 4]], np.float32)
    expected_targets = np.empty((3, 3, 2), np.float32)
    for b in range(3):
        for w, weight in enumerate(weights):
            action = np.argmax((online * weight).sum(-1))
            expected_targets[b, w] = (
                np.array(batch.rewards[b])
                + 0.9
                * (1 - float(batch.extras.env_extras.termination[b]))
                * target[action]
            )
    current = np.broadcast_to(online[np.array(batch.actions)][:, None, :], (3, 3, 2))
    errors = current - expected_targets
    vector_loss = (errors**2).sum(-1).mean()
    scalar_loss = np.abs((errors * weights).sum(-1)).mean()
    fn = lambda state: agent.loss(state, batch, key)
    actual = (jax.jit(fn) if jit else fn)(state)
    np.testing.assert_allclose(actual.vector_loss, vector_loss, rtol=2e-6)
    np.testing.assert_allclose(actual.scalar_loss, scalar_loss, rtol=2e-6)
    np.testing.assert_allclose(
        actual.q_loss, (1 - mix) * vector_loss + mix * scalar_loss, rtol=2e-6
    )

    models = agent.get_models(state, copy=True)
    grads = nnx.grad(
        lambda models: (
            agent.loss(state.replace(model_state=nnx.unpack(models)), batch, key).q_loss
        )
    )(models)
    for value in jax.tree.leaves(grads["target_q"]):
        np.testing.assert_array_equal(value, 0)


def test_envelope_inference_responds_to_explicit_preferences():
    agent, state = envelope_q_agent()
    batch = SampleBatch(obs=jnp.ones((2, 3)), extras=PyTreeDict(preferences=jnp.eye(2)))
    actions, _ = agent.evaluate_actions(state, batch, jax.random.key(0))
    np.testing.assert_array_equal(actions, [0, 1])
    # A default inference preference in AgentState follows the same API.
    state = state.replace(
        extra_state=state.extra_state.replace(preferences=jnp.array([0.0, 1.0]))
    )
    actions, _ = agent.evaluate_actions(
        state, SampleBatch(obs=jnp.ones((2, 3))), jax.random.key(0)
    )
    np.testing.assert_array_equal(actions, [1, 1])


def capql_agent(alpha=0.2, objectives=2):
    agent = CAPQLAgent(
        num_objectives=objectives,
        agent_network_cfg=OmegaConf.create(
            dict(
                actor_hidden_layer_sizes=[], critic_hidden_layer_sizes=[], num_critics=2
            )
        ),
        discount=0.9,
        alpha=alpha,
    )
    state = agent.init(
        Box(low=-jnp.ones(3), high=jnp.ones(3)),
        Box(low=-jnp.ones(2), high=jnp.ones(2)),
        jax.random.key(0),
    )
    models = agent.get_models(state, copy=True)
    nnx.update(models, jax.tree.map(jnp.zeros_like, nnx.state(models, nnx.Param)))
    models.critic.mlp.mlp.layers[0].bias[...] = jnp.array([[2.0, 9.0], [8.0, 3.0]])[
        :, :objectives
    ]
    models.target_critic.mlp.mlp.layers[0].bias[...] = jnp.array(
        [[5.0, 8.0], [7.0, 2.0]]
    )[:, :objectives]
    return agent, state.replace(model_state=nnx.unpack(models))


@pytest.mark.parametrize("objectives", [1, 2])
def test_capql_vector_target_entropy_and_losses_match_density_calculation(
    objectives, jit
):
    agent, state = capql_agent(objectives=objectives)
    weights = (
        jnp.array([[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]])
        if objectives == 2
        else jnp.ones((3, 1))
    )
    rewards = jnp.array([[1.0, 2.0], [-1.0, 3.0], [2.0, -2.0]])[:, :objectives]
    batch = SampleBatch(
        obs=jnp.ones((3, 3)),
        actions=jnp.zeros((3, 2)),
        rewards=rewards,
        dones=jnp.array([1.0, 1.0, 0.0]),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=jnp.ones((3, 3)) * 2,
                termination=jnp.array([1.0, 0.0, 0.0]),
            ),
            preferences=weights,
        ),
    )
    key = jax.random.key(12)
    # Independent Gaussian density + tanh Jacobian in normalized coordinates. Networks
    # have zero parameters, so mean=0 and std=softplus(0)+1e-3.
    normalized = np.array(
        get_tanh_norm_dist(jnp.zeros((3, 2)), jnp.zeros((3, 2))).sample(seed=key)
    )
    std = np.log(2) + 0.001
    z = np.arctanh(normalized)
    logp = (
        -0.5 * (z / std) ** 2
        - np.log(std)
        - 0.5 * np.log(2 * np.pi)
        - np.log1p(-(normalized**2))
    ).sum(-1)
    next_q = np.array([5.0, 2.0])[:objectives]
    targets = np.array(rewards) + 0.9 * (
        1 - np.array(batch.extras.env_extras.termination)
    )[:, None] * (next_q - 0.2 * logp[:, None])
    online = np.array([[2.0, 9.0], [8.0, 3.0]])[:, :objectives]
    expected_critic = ((online[None, :, :] - targets[:, None, :]) ** 2).sum(1).mean()
    expected_actor = (
        0.2 * logp - (np.array(weights) * np.array([2.0, 3.0])[:objectives]).sum(-1)
    ).mean()
    critic_fn = lambda state: agent.critic_loss(state, batch, key)
    actor_fn = lambda state: agent.actor_loss(state, batch, key)
    critic = (jax.jit(critic_fn) if jit else critic_fn)(state)
    actor = (jax.jit(actor_fn) if jit else actor_fn)(state)
    np.testing.assert_allclose(critic.critic_loss, expected_critic, rtol=3e-6)
    np.testing.assert_allclose(actor.actor_loss, expected_actor, rtol=3e-6)


def test_capql_actor_gradient_matches_reparameterized_weighted_vector_critic():
    agent, state = capql_agent(alpha=0.0)
    models = agent.get_models(state, copy=True)
    kernel = jnp.zeros_like(models.critic.mlp.mlp.layers[0].kernel[...])
    action_weights = jnp.array([[1.0, 4.0], [2.0, 3.0]])
    kernel = kernel.at[:, -2:, :].set(action_weights)
    models.critic.mlp.mlp.layers[0].kernel[...] = kernel
    models.critic.mlp.mlp.layers[0].bias[...] = jnp.array([[0.0, 0.0], [10.0, 10.0]])
    state = state.replace(model_state=nnx.unpack(models))
    weights = jnp.array([[0.25, 0.75], [0.8, 0.2], [0.5, 0.5]])
    batch = SampleBatch(obs=jnp.ones((3, 3)), extras=PyTreeDict(preferences=weights))
    key = jax.random.key(2)
    normalized = get_tanh_norm_dist(jnp.zeros((3, 2)), jnp.zeros((3, 2))).sample(
        seed=key
    )
    standard_noise = jax.lax.stop_gradient(
        jnp.arctanh(normalized) / (jnp.log(2.0) + 0.001)
    )

    def reference(raw):
        mean, raw_scale = jnp.split(raw, 2)
        action = jnp.tanh(mean + (jax.nn.softplus(raw_scale) + 0.001) * standard_noise)
        return -((action @ action_weights) * weights).sum(-1).mean()

    expected = jax.grad(reference)(jnp.zeros(4))
    grads = nnx.grad(
        lambda models: (
            agent.actor_loss(
                state.replace(model_state=nnx.unpack(models)), batch, key
            ).actor_loss
        ),
        argnums=nnx.DiffState(0, nnx.All(nnx.Param, nnx.PathContains("actor"))),
    )(models)
    np.testing.assert_allclose(
        grads["actor"]["mlp"]["layers"][0]["bias"][...], expected, atol=2e-6, rtol=2e-6
    )
    assert set(grads) == {"actor"}
