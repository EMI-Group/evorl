"""Native NNX policy/value training with explicit, independently owned state."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from omegaconf import OmegaConf

from evorl.algorithms.meta.pbt_ppo.param_ppo import ParamPPOAgent
from evorl.algorithms.a2c import A2CAgent
from evorl.algorithms.ec.ec_agent import DeterministicECAgent, StochasticECAgent
from evorl.algorithms.impala import IMPALAAgent
from evorl.algorithms.ppo import PPOAgent
from evorl.distributed import pmean_nnx_loss_and_grad
from evorl.envs import Box, SpaceContainer
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils import running_statistics

from tests.helpers.agents import build_agent
from tests.helpers.envs import ToyEnv


FACTORIES = {
    "a2c": partial(build_agent, A2CAgent),
    "ppo": partial(build_agent, PPOAgent),
    "impala": partial(build_agent, IMPALAAgent),
}


def make_batch(agent, state, name):
    """Construct one fixed on-policy batch with behavior log probabilities."""
    obs = jax.random.normal(jax.random.key(2), (3, 4, 3)) + 5
    actions, policy_extras = agent.compute_actions(
        state, SampleBatch(obs=obs), jax.random.key(3)
    )
    batch = SampleBatch(
        obs=obs,
        actions=actions,
        next_obs=obs + 0.1,
        rewards=jnp.ones((3, 4)),
        dones=jnp.zeros((3, 4)),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                autoreset=jnp.zeros((3, 4)), termination=jnp.zeros((3, 4))
            ),
            policy_extras=policy_extras,
            v_targets=jnp.ones((3, 4)),
            advantages=jnp.linspace(-1, 1, 12).reshape(3, 4),
        ),
    )
    if name != "impala":
        batch = jax.tree.map(lambda x: x.reshape(12, *x.shape[2:]), batch)
    return batch


@pytest.mark.parametrize("name", FACTORIES)
@pytest.mark.parametrize("discrete", [True, False])
def test_nnx_joint_update_matches_explicit_parameter_gradient(name, discrete, jit):
    """NNX differentiation and Adam preserve the original joint-loss update."""
    env = ToyEnv(4, discrete=discrete)
    agent = FACTORIES[name](
        env.obs_space,
        env.action_space,
        actor_hidden_layer_sizes=(4,),
        critic_hidden_layer_sizes=(4,),
    )
    static_fields = vars(agent).copy()
    original_hash = hash(agent)
    state = agent.init(env.obs_space, env.action_space, jax.random.key(0))
    before = jax.tree.map(lambda x: np.asarray(x).copy(), state.model_state)
    batch = make_batch(agent, state, name)
    key = jax.random.key(7)
    tx = optax.adam(1e-3)
    models = nnx.merge(state.model_state, copy=True)
    optimizer_state = nnx.unpack(nnx.Optimizer(models, tx, wrt=nnx.Param))

    def objective(models):
        losses = agent.loss(state.replace(model_state=nnx.unpack(models)), batch, key)
        return (
            losses.actor_loss + 0.5 * losses.critic_loss - 0.01 * losses.actor_entropy
        )

    def pure_objective(params):
        return objective(nnx.merge(params, copy=True))

    expected_loss, grads = jax.value_and_grad(pure_objective)(state.model_state)
    updates, _ = tx.update(grads, tx.init(state.model_state), state.model_state)
    expected_params = optax.apply_updates(state.model_state, updates)

    def update(params, opt_state):
        models = nnx.merge(params, copy=True)
        optimizer = nnx.merge(opt_state, copy=True)
        loss, grads = pmean_nnx_loss_and_grad(objective, None)(models)
        optimizer.update(models, grads)
        return loss, nnx.unpack(models), nnx.unpack(optimizer)

    if jit:
        update = jax.jit(update)
    loss, params, new_optimizer_state = update(state.model_state, optimizer_state)
    np.testing.assert_allclose(loss, expected_loss, atol=1e-6)
    for actual, expected in zip(
        jax.tree.leaves(params), jax.tree.leaves(expected_params), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    for actual, original in zip(
        jax.tree.leaves(state.model_state), jax.tree.leaves(before), strict=True
    ):
        np.testing.assert_array_equal(actual, original)
    assert int(nnx.merge(new_optimizer_state, copy=True).step[...]) == 1
    assert vars(agent) == static_fields
    assert hash(agent) == original_hash


@pytest.mark.parametrize("name", FACTORIES)
def test_loss_normalizes_actor_and_value_observations_consistently(name):
    """Unchanged normalized IMPALA policies must have importance ratio one."""
    env = ToyEnv(4, discrete=True)
    agent = FACTORIES[name](
        env.obs_space,
        env.action_space,
        actor_hidden_layer_sizes=(4,),
        critic_hidden_layer_sizes=(4,),
        normalize_obs=True,
    )
    state = agent.init(env.obs_space, env.action_space, jax.random.key(0))
    stats = running_statistics.update(
        state.obs_preprocessor_state,
        jax.random.normal(jax.random.key(6), (32, 3)) + 5,
    )
    state = state.replace(obs_preprocessor_state=stats)
    batch = make_batch(agent, state, name)
    normalized_batch = batch.replace(
        obs=running_statistics.normalize(batch.obs, stats),
        next_obs=running_statistics.normalize(batch.next_obs, stats),
    )
    actual = agent.loss(state, batch, jax.random.key(7))
    unnormalized_agent = agent.replace(obs_preprocessor=None)
    expected = unnormalized_agent.loss(state, normalized_batch, jax.random.key(7))
    for loss_name in actual:
        np.testing.assert_allclose(actual[loss_name], expected[loss_name], atol=1e-6)
    if name == "impala":
        np.testing.assert_allclose(actual.rho, 1, atol=1e-6)


@pytest.mark.parametrize(
    "factory",
    [
        partial(build_agent, StochasticECAgent),
        partial(build_agent, DeterministicECAgent),
    ],
)
@pytest.mark.parametrize("norm", ["none", "layer_norm", "spectral_norm"])
def test_ec_model_state_preserves_variable_types_and_supports_nested_vmap(
    factory, norm
):
    env = ToyEnv(4)
    agent = factory(
        env.obs_space,
        env.action_space,
        actor_hidden_layer_sizes=(4,),
        norm_layer_type=norm,
    )
    keys = jax.random.split(jax.random.key(1), 6).reshape(2, 3)
    init = lambda key: agent.init(env.obs_space, env.action_space, key)
    states = jax.jit(jax.vmap(jax.vmap(init)))(keys)
    obs = jnp.ones((4, 3))
    act = lambda state, key: agent.compute_actions(state, SampleBatch(obs=obs), key)[0]
    actual = jax.jit(jax.vmap(jax.vmap(act)))(states, keys)
    single = init(keys[0, 0])
    expected = act(single, keys[0, 0])
    np.testing.assert_allclose(actual[0, 0], expected, atol=1e-6)
    assert actual.shape == (2, 3, 4, 1)
    models = nnx.merge(single.model_state, copy=True)
    assert jax.tree.leaves(nnx.state(models, nnx.Param))
    assert bool(jax.tree.leaves(nnx.state(models, nnx.Not(nnx.Param)))) == (
        norm == "spectral_norm"
    )


def test_paramppo_uses_runtime_clipping_without_changing_static_agent():

    env = ToyEnv(4, discrete=True)
    agent = build_agent(
        ParamPPOAgent,
        env.obs_space,
        env.action_space,
        actor_hidden_layer_sizes=(4,),
        critic_hidden_layer_sizes=(4,),
        normalize_gae=True,
    )
    state = agent.init(env.obs_space, env.action_space, jax.random.key(0))
    batch = make_batch(agent, state, "ppo")
    batch.extras.policy_extras.logp -= jnp.log(2.0)

    @jax.jit
    def loss(epsilon):
        candidate = state.replace(
            extra_state=state.extra_state.replace(clip_epsilon=epsilon)
        )
        return agent.loss(candidate, batch, jax.random.key(7)).actor_loss

    advantages = batch.extras.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for epsilon in (0.0, 0.2, 0.5):
        expected = -jnp.minimum(2 * advantages, (1 + epsilon) * advantages).mean()
        np.testing.assert_allclose(loss(jnp.float32(epsilon)), expected, atol=1e-6)
    assert agent.clip_epsilon == 0.2


@pytest.mark.parametrize("name", FACTORIES)
def test_network_config_supports_distinct_observation_fields_and_nested_vmap(name):
    """The static config builds each model; only its variables gain population axes."""
    obs_space = SpaceContainer(
        spaces=dict(
            actor=Box(low=-jnp.ones(3), high=jnp.ones(3)),
            critic=Box(low=-jnp.ones(5), high=jnp.ones(5)),
        )
    )
    action_space = ToyEnv(2).action_space
    network = OmegaConf.create(
        dict(
            actor_hidden_layer_sizes=[4],
            critic_hidden_layer_sizes=[6],
            policy_obs_key="actor",
            value_obs_key="critic",
        )
    )
    agent = FACTORIES[name](obs_space, action_space, agent_network_cfg=network)
    keys = jax.random.split(jax.random.key(20), 6).reshape(2, 3)
    init = lambda key: agent.init(obs_space, action_space, key)
    state = init(keys[0, 0])
    models = nnx.merge(state.model_state, copy=True)
    assert models.policy.mlp.layers[0].kernel.shape == (3, 4)
    assert models.value.mlp.layers[0].kernel.shape == (5, 6)

    states = jax.jit(jax.vmap(jax.vmap(init)))(keys)
    batch = SampleBatch(obs=dict(actor=jnp.ones((2, 3)), critic=jnp.ones((2, 5))))
    act = lambda state, key: agent.compute_actions(state, batch, key)[0]
    actions = jax.jit(jax.vmap(jax.vmap(act)))(states, keys)
    expected = act(state, keys[0, 0])
    np.testing.assert_allclose(actions[0, 0], expected, atol=1e-6)
    assert actions.shape == (2, 3, 2, 1)
