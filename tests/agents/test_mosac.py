"""MOSAC scalarization, soft Bellman targets and entropy tuning formulas."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.agent import AgentState
from evorl.algorithms.multi_objective.mosac import (
    MOSACAgent,
    scalarized_q,
)
from evorl.algorithms.sac import SACAgent, SACDiscreteAgent
from evorl.distribution import get_tanh_norm_dist
from evorl.distributed import pmean_nnx_loss_and_grad
from evorl.envs.wrappers import ActionSquashWrapper
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from tests.helpers.multi_objective_envs import ToyMultiObjectiveEnv


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("component", ["critic", "actor", "alpha"])
def test_single_objective_losses_and_selected_gradients_match_sac(
    continuous, component
):
    env = ToyMultiObjectiveEnv(continuous, num_objectives=1)
    if continuous:
        env = ActionSquashWrapper(env)
    cfg = OmegaConf.create(
        dict(num_critics=2, actor_hidden_layer_sizes=[4], critic_hidden_layer_sizes=[4])
    )
    sac_cls = SACAgent if continuous else SACDiscreteAgent
    sac = sac_cls(agent_network_cfg=cfg, init_alpha=0.2, discount=0.9)
    mosac = MOSACAgent(
        num_objectives=1,
        continuous_action=continuous,
        agent_network_cfg=cfg,
        init_alpha=0.2,
        discount=0.9,
    )
    key = jax.random.key(29)
    sac_state = sac.init(env.obs_space, env.action_space, key)
    mosac_state = mosac.init(env.obs_space, env.action_space, key)
    for scalar, vector in zip(
        jax.tree.leaves(sac_state.model_state),
        jax.tree.leaves(mosac_state.model_state),
        strict=True,
    ):
        np.testing.assert_array_equal(scalar, vector)
    obs = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
    actions, _ = sac.compute_actions(sac_state, SampleBatch(obs=obs), key)
    mo_actions, _ = mosac.compute_actions(mosac_state, SampleBatch(obs=obs), key)
    np.testing.assert_array_equal(actions, mo_actions)
    batch = SampleBatch(
        obs=obs,
        actions=actions,
        rewards=jnp.array([1.0, -1.0, 0.5, 2.0]),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=obs + 0.2, termination=jnp.array([0.0, 1.0, 0.0, 1.0])
            )
        ),
    )
    results = []
    for agent, state, samples in (
        (sac, sac_state, batch),
        (mosac, mosac_state, batch.replace(rewards=batch.rewards[:, None])),
    ):
        loss_fn = getattr(agent, f"{component}_loss")

        def objective(models, loss_fn=loss_fn, state=state, samples=samples):
            return loss_fn(state.replace(model_state=nnx.unpack(models)), samples, key)[
                f"{component}_loss"
            ]

        results.append(
            pmean_nnx_loss_and_grad(
                objective,
                None,
                wrt=nnx.All(
                    nnx.Param,
                    nnx.PathContains(
                        "log_alpha" if component == "alpha" else component
                    ),
                ),
            )(agent.get_models(state, copy=True))
        )
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5, atol=1e-6)
    for scalar, vector in zip(
        jax.tree.leaves(results[0][1]), jax.tree.leaves(results[1][1]), strict=True
    ):
        np.testing.assert_allclose(scalar, vector, rtol=1e-5, atol=1e-6)


class ConstantNetwork(nnx.Module):
    def __init__(self, value):
        self.value = nnx.Param(jnp.asarray(value, jnp.float32))

    def __call__(self, obs, actions=None):
        return jnp.broadcast_to(self.value[...], (*obs.shape[:-1], *self.value.shape))


def discrete_fixture():
    agent = MOSACAgent(
        num_objectives=2,
        continuous_action=False,
        discount=0.5,
        agent_network_cfg=OmegaConf.create(dict(num_critics=2)),
        init_alpha=0.2,
    )
    q = jnp.array([[[0.0, 10.0], [4.0, 2.0]], [[10.0, 0.0], [2.0, 4.0]]])
    models = nnx.Dict(
        actor=ConstantNetwork([0.0, 0.0]),
        critic=ConstantNetwork(q),
        target_critic=ConstantNetwork(q),
        log_alpha=nnx.Param(jnp.log(jnp.float32(0.2))),
    )
    state = AgentState(
        model_state=nnx.unpack(models),
        extra_state=PyTreeDict(
            preferences=jnp.array([0.5, 0.5]), target_entropy=0.98 * jnp.log(2.0)
        ),
    )
    batch = SampleBatch(
        obs=jnp.zeros((2, 3)),
        actions=jnp.zeros(2, jnp.int32),
        rewards=jnp.array([[2.0, 4.0], [2.0, 4.0]]),
        dones=jnp.ones(2),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=jnp.zeros((2, 3)), termination=jnp.array([1.0, 0.0])
            )
        ),
    )
    return agent, state, batch


def test_scalarization_precedes_critic_minimum():
    q = jnp.array([[0.0, 10.0], [10.0, 0.0]])
    np.testing.assert_allclose(scalarized_q(q, jnp.array([0.5, 0.5])).min(), 5.0)
    assert float((q.min(0) * 0.5).sum()) == 0.0


def test_discrete_bellman_bootstraps_truncation_and_scalarizes_before_minimum():
    agent, state, batch = discrete_fixture()
    loss = jax.jit(agent.critic_loss)(state, batch, jax.random.key(0))
    next_value = 4 + 0.2 * np.log(2)
    targets = np.array([3.0, 3.0 + 0.5 * next_value])
    np.testing.assert_allclose(loss.target_value, targets.mean(), rtol=1e-6)
    np.testing.assert_allclose(
        loss.critic_loss, (2 * np.square(5.0 - targets)).mean(), rtol=1e-6
    )
    actor = agent.actor_loss(state, batch, jax.random.key(1))
    np.testing.assert_allclose(actor.actor_loss, -4.0 - 0.2 * np.log(2), rtol=1e-6)
    np.testing.assert_allclose(actor.entropy, np.log(2), rtol=1e-6)


def test_discrete_temperature_gradient_tracks_entropy_shortfall():
    agent, state, batch = discrete_fixture()

    def loss(log_alpha, logits):
        models = agent.get_models(state, copy=True)
        models.log_alpha[...] = log_alpha
        models.actor.value[...] = logits
        return agent.alpha_loss(
            state.replace(model_state=nnx.unpack(models)), batch, jax.random.key(2)
        ).alpha_loss

    grad = jax.grad(loss)
    assert float(grad(jnp.log(0.2), jnp.zeros(2))) > 0  # decrease alpha above target
    assert (
        float(grad(jnp.log(0.2), jnp.array([8.0, -8.0]))) < 0
    )  # increase below target


@pytest.mark.parametrize("continuous", [True, False])
def test_networks_are_observation_only_and_preferences_do_not_change_actions(
    continuous,
):
    env = ToyMultiObjectiveEnv(continuous)
    if continuous:
        env = ActionSquashWrapper(env)
    agent = MOSACAgent(
        num_objectives=2,
        continuous_action=continuous,
        agent_network_cfg=OmegaConf.create(
            dict(
                num_critics=2,
                actor_hidden_layer_sizes=[4],
                critic_hidden_layer_sizes=[4],
            )
        ),
    )
    state = agent.init(env.obs_space, env.action_space, jax.random.key(3))
    batch = SampleBatch(obs=jnp.ones((8, 3)))
    switched = state.replace(
        extra_state=state.extra_state.replace(preferences=jnp.array([1.0, 0.0]))
    )
    first, _ = agent.compute_actions(state, batch, jax.random.key(4))
    second, _ = agent.compute_actions(switched, batch, jax.random.key(4))
    np.testing.assert_array_equal(first, second)
    if continuous:
        assert bool(jnp.all(first >= env.action_space.low))
        assert bool(jnp.all(first <= env.action_space.high))
    models = agent.get_models(state)
    assert models.actor.mlp.layers[0].kernel.shape[0] == 3
    values = models.critic(batch.obs, first) if continuous else models.critic(batch.obs)
    assert values.shape == ((8, 2, 2) if continuous else (8, 2, 2, 2))


def test_continuous_sac_distribution_and_normalized_action_density():
    env = ActionSquashWrapper(ToyMultiObjectiveEnv(True))
    agent = MOSACAgent(
        num_objectives=2,
        continuous_action=True,
        discount=0.5,
        agent_network_cfg=OmegaConf.create(
            dict(
                num_critics=2,
                actor_hidden_layer_sizes=[4],
                critic_hidden_layer_sizes=[4],
            )
        ),
        init_alpha=0.2,
    )
    state = agent.init(env.obs_space, env.action_space, jax.random.key(10))
    raw = jnp.zeros((2, 4))
    models = nnx.Dict(
        actor=ConstantNetwork(jnp.zeros(4)),
        critic=ConstantNetwork([[0.0, 10.0], [10.0, 0.0]]),
        target_critic=ConstantNetwork([[0.0, 10.0], [10.0, 0.0]]),
        log_alpha=nnx.Param(jnp.log(jnp.float32(0.2))),
    )
    state = state.replace(model_state=nnx.unpack(models))
    batch = SampleBatch(
        obs=jnp.zeros((2, 3)),
        actions=jnp.zeros((2, 2)),
        rewards=jnp.array([[2.0, 4.0], [2.0, 4.0]]),
        dones=jnp.ones(2),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=jnp.zeros((2, 3)), termination=jnp.array([1.0, 0.0])
            )
        ),
    )
    dist = get_tanh_norm_dist(*jnp.split(raw, 2, -1))
    np.testing.assert_allclose(
        dist.distribution.distribution.stddev(), np.log(2) + 1e-3, rtol=1e-6
    )
    key = jax.random.key(11)
    sampled = dist.sample(seed=key)
    logp = np.asarray(dist.log_prob(sampled))
    assert "action_scale" not in state.extra_state
    assert "action_bias" not in state.extra_state
    target = np.array([3.0, 3.0 + 0.5 * (5.0 - 0.2 * logp[1])])
    actual = agent.critic_loss(state, batch, key)
    np.testing.assert_allclose(actual.target_value, target.mean(), rtol=1e-6)
    np.testing.assert_allclose(
        actual.critic_loss, (2 * np.square(5.0 - target)).mean(), rtol=1e-6
    )
    actor = agent.actor_loss(state, batch, key)
    actor_key, _ = jax.random.split(key)
    actor_logp = dist.log_prob(dist.sample(seed=actor_key))
    np.testing.assert_allclose(
        actor.actor_loss, (0.2 * actor_logp - 5.0).mean(), rtol=1e-6
    )
