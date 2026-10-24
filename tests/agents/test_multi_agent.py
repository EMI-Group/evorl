"""MARL targets, policy gradients and masks against explicit references."""

from functools import partial

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from evorl.algorithms.multi_agents.common import agent_layout, masked_mean
from evorl.algorithms.multi_agents.ippo import IPPOAgent, compute_agent_gae
from evorl.algorithms.multi_agents.maddpg import MADDPGAgent
from evorl.algorithms.multi_agents.matd3 import MATD3Agent
from evorl.rollout import multi_agent_rollout
from evorl.distributed import pmean_nnx_loss_and_grad
from evorl.envs.wrappers import (
    EpisodeWrapper,
    VmapAutoResetWrapper,
    VmapEnvPoolAutoResetWrapper,
)
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.rl_toolkits import flatten_rollout_trajectory
from tests.helpers.multi_agent_envs import ToyMultiAgentEnv


def make_agent(cls, continuous=True, **kwargs):
    env = VmapAutoResetWrapper(EpisodeWrapper(ToyMultiAgentEnv(continuous), 3), 3)
    agent = cls(
        **agent_layout(env),
        agent_network_cfg=OmegaConf.create(
            dict(actor_hidden_layer_sizes=[], critic_hidden_layer_sizes=[])
        ),
        **kwargs,
    )
    state = agent.init(env.obs_space, env.action_space, jax.random.key(0))
    return env, agent, state


@pytest.mark.parametrize("centralized", [False, True])
def test_envpool_agent_gae_keeps_terminal_bootstrap_and_skips_reset(centralized, jit):
    env = VmapEnvPoolAutoResetWrapper(
        EpisodeWrapper(ToyMultiAgentEnv(True), 3, record_ori_obs=False), 2
    )
    agent = IPPOAgent(
        **agent_layout(env),
        centralized_critic=centralized,
        agent_network_cfg=OmegaConf.create(
            dict(actor_hidden_layer_sizes=[], critic_hidden_layer_sizes=[])
        ),
    )
    state = agent.init(env.obs_space, env.action_space, jax.random.key(31))
    models = agent.get_models(state, copy=True)
    for name in agent.agents:
        layer = models[name].value.mlp.layers[0]
        layer.kernel[...] = jnp.zeros_like(layer.kernel[...]).at[0, 0].set(1)
        layer.bias[...] = jnp.zeros_like(layer.bias[...])
    state = state.replace(model_state=nnx.unpack(models))

    def collect(key):
        trajectory, final = multi_agent_rollout(
            env, agent.compute_actions, env.reset(key), state, key, 6
        )
        obs = jax.tree.map(
            lambda x, last: jnp.concatenate([x, last[None]]),
            trajectory.obs,
            final.obs,
        )
        global_obs = jnp.concatenate(
            [trajectory.extras.global_obs, final.info.global_obs[None]]
        )
        values = agent.compute_values(
            state, SampleBatch(obs=obs, extras=PyTreeDict(global_obs=global_obs))
        )
        result = PyTreeDict()
        for name in agent.agents:
            result[name] = compute_agent_gae(
                trajectory.rewards[name],
                values[name][:-1],
                values[name][1:],
                trajectory.dones[name],
                trajectory.extras.terminations[name],
                trajectory.extras.active[name],
                0.5,
                1.0,
            )
        return result, trajectory.extras.active

    result, active = (jax.jit(collect) if jit else collect)(jax.random.key(32))
    # First dies at step 2: reward only. Second truncates at step 3:
    # -2 + 0.5 * V(terminal count=3) = -0.5, rather than V(reset count=0).
    np.testing.assert_allclose(result.first[0][1], 1)
    np.testing.assert_allclose(result.second[0][2], -0.5)
    for name in agent.agents:
        np.testing.assert_array_equal(active[name][3], False)
        np.testing.assert_array_equal(result[name][1][3], 0)


@pytest.mark.parametrize("lambda_", [0.0, 0.9, 1.0])
def test_agent_gae_matches_numpy_at_death_truncation_and_rollout_boundary(lambda_, jit):
    rewards = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], np.float32)
    values = np.arange(8, dtype=np.float32).reshape(4, 2) / 10
    next_values = values + 0.3
    # Both types of episode boundary are present; truncation has a large
    # terminal value which must not be replaced with a reset observation value.
    next_values[1, 1] = 20
    dones = np.array([[0, 0], [1, 1], [1, 0], [1, 0]], np.float32)
    terminations = np.array([[0, 0], [1, 0], [1, 0], [1, 0]], np.float32)
    active = np.array([[1, 1], [1, 1], [0, 1], [0, 1]], bool)
    expected = np.zeros_like(values)
    carry = np.zeros(2, np.float32)
    for t in reversed(range(4)):
        delta = rewards[t] + 0.95 * (1 - terminations[t]) * next_values[t] - values[t]
        carry = np.where(active[t], delta + 0.95 * lambda_ * (1 - dones[t]) * carry, 0)
        expected[t] = carry
    target_fn = partial(compute_agent_gae, discount=0.95, gae_lambda=lambda_)
    targets, advantages = (jax.jit(target_fn) if jit else target_fn)(
        *map(jnp.asarray, (rewards, values, next_values, dones, terminations, active))
    )
    np.testing.assert_allclose(advantages, expected, atol=2e-6, rtol=1e-6)
    np.testing.assert_allclose(targets, expected + values, atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize("cls", [MADDPGAgent, MATD3Agent])
def test_centralized_critic_targets_use_own_reward_and_termination(cls, jit):
    env, agent, state = make_agent(cls, discount=0.9)
    models = agent.get_models(state, copy=True)
    for name in agent.agents:
        for network in (models[name].critic, models[name].target_critic):
            nnx.update(
                network, jax.tree.map(jnp.zeros_like, nnx.state(network, nnx.Param))
            )
        online = [2] if cls is MADDPGAgent else [[2], [3]]
        target = [5] if name == "first" else [-4]
        if cls is MATD3Agent:
            target = [[target[0]], [target[0] + 2]]
        critic_mlp = models[name].critic.mlp
        target_mlp = models[name].target_critic.mlp
        if cls is MATD3Agent:
            critic_mlp, target_mlp = critic_mlp.mlp, target_mlp.mlp
        critic_mlp.layers[-1].bias[...] = jnp.array(online, jnp.float32)
        target_mlp.layers[-1].bias[...] = jnp.array(target, jnp.float32)
    state = state.replace(model_state=nnx.unpack(models))
    trajectory, _ = multi_agent_rollout(
        env,
        agent.compute_actions,
        env.reset(jax.random.key(1)),
        state,
        jax.random.key(2),
        1,
    )
    batch = flatten_rollout_trajectory(trajectory).replace(
        rewards=PyTreeDict(
            first=jnp.array([1.0, 2.0, 3.0]), second=jnp.array([-1.0, 2.0, -3.0])
        ),
        dones=PyTreeDict(
            first=jnp.array([0.0, 1.0, 1.0]), second=jnp.array([0.0, 0.0, 1.0])
        ),
    )
    batch = batch.replace(
        extras=batch.extras.replace(
            terminations=PyTreeDict(
                first=jnp.array([0.0, 1.0, 0.0]), second=jnp.array([0.0, 0.0, 1.0])
            ),
            active=PyTreeDict(
                first=jnp.array([True, True, True]),
                second=jnp.array([True, False, True]),
            ),
        )
    )
    for name, targets, selected in (
        ("first", [5.5, 2, 7.5], [True, True, True]),
        ("second", [-4.6, -1.6, -3], [True, False, True]),
    ):
        loss_fn = partial(agent.critic_loss, agent_id=name)
        loss = (jax.jit(loss_fn) if jit else loss_fn)(state, batch, jax.random.key(3))
        online_q = np.array([2] if cls is MADDPGAgent else [2, 3])
        errors = (online_q[None, :] - np.array(targets)[:, None]) ** 2
        expected = errors.sum(axis=-1)[selected].mean()
        np.testing.assert_allclose(loss.critic_loss, expected, rtol=1e-6)


@pytest.mark.parametrize("cls", [MADDPGAgent, MATD3Agent])
def test_actor_gradient_changes_only_own_action_and_uses_first_critic(cls, jit):
    env, agent, state = make_agent(cls)
    models = agent.get_models(state, copy=True)
    nnx.update(
        models.first.actor,
        jax.tree.map(jnp.zeros_like, nnx.state(models.first.actor, nnx.Param)),
    )
    weights = jnp.array([0, 0, 2, 3, 7], jnp.float32)[:, None]
    critic_mlp = models.first.critic.mlp
    if cls is MATD3Agent:
        critic_mlp = critic_mlp.mlp
    critic_mlp.layers[0].kernel[...] = (
        weights if cls is MADDPGAgent else jnp.stack([weights, 10 * weights])
    )
    critic_mlp.layers[0].bias[...] = jnp.zeros_like(critic_mlp.layers[0].bias[...])
    state = state.replace(model_state=nnx.unpack(models))
    trajectory, _ = multi_agent_rollout(
        env,
        agent.compute_actions,
        env.reset(jax.random.key(1)),
        state,
        jax.random.key(2),
        1,
    )
    batch = flatten_rollout_trajectory(trajectory)

    def gradient(state):
        models = agent.get_models(state, copy=True)

        def loss_fn(models):
            return agent.actor_loss(
                state.replace(model_state=nnx.unpack(models)),
                batch,
                jax.random.key(0),
                "first",
            ).actor_loss

        return nnx.grad(
            loss_fn,
            argnums=nnx.DiffState(
                0,
                lambda path, variable: (
                    path[:2] == ("first", "actor") and isinstance(variable, nnx.Param)
                ),
            ),
        )(models)

    grads = (jax.jit(gradient) if jit else gradient)(state)
    bias_grad = grads["first"]["actor"]["mlp"]["layers"][0]["bias"][...]
    # Physical bounds [-2, 2] multiply dQ/da=(2,3) by two; tanh'(0)=1.
    np.testing.assert_allclose(bias_grad, [-4, -6], atol=1e-6)
    assert set(grads) == {"first"}


def test_discrete_policies_respect_masks_under_nested_vmap(jit):
    env, agent, state = make_agent(IPPOAgent, continuous=False)
    native = env.unwrapped
    obs = jax.vmap(native._observations)(jnp.arange(3))
    batch = SampleBatch(obs=obs)
    keys = jax.random.split(jax.random.key(5), (2, 2))
    action_fn = jax.vmap(jax.vmap(lambda key: agent.compute_actions(state, batch, key)))
    actions, extras = (jax.jit(action_fn) if jit else action_fn)(keys)
    for name, count in [("first", 2), ("second", 3)]:
        np.testing.assert_array_equal(
            actions[name], jnp.broadcast_to(jnp.arange(3) % count, (2, 2, 3))
        )
        np.testing.assert_allclose(extras.logp[name], 0)


def test_global_masked_mean_matches_unbalanced_valid_sample_counts(dp_mesh):
    state = nnx.unpack(nnx.Dict(weight=nnx.Param(jnp.float32(2))))
    values = jax.device_put(jnp.array([1.0, 10.0, 100.0, 2.0, 3.0, 4.0]), jax.P("DP"))
    masks = jax.device_put(
        jnp.array([True, False, False, True, True, True]), jax.P("DP")
    )

    @jax.shard_map(
        in_specs=(jax.P(), jax.P("DP"), jax.P("DP")), out_specs=(jax.P(), jax.P())
    )
    def loss_and_grad(state, values, masks):
        models = nnx.merge(state, copy=True)
        return pmean_nnx_loss_and_grad(
            lambda model: masked_mean(model.weight[...] * values, masks, "DP"), "DP"
        )(models)

    loss, grads = jax.jit(loss_and_grad)(state, values, masks)
    np.testing.assert_allclose(loss, 5)
    np.testing.assert_allclose(grads["weight"][...], 2.5)
