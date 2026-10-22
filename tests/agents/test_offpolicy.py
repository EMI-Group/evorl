"""Functional NNX training contracts for off-policy RL agents."""

from functools import partial
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from evorl.algorithms.ddpg import DDPGAgent
from evorl.algorithms.dqn import DQNAgent
from evorl.algorithms.sac import SACAgent
from evorl.algorithms.td3 import TD3Agent
from evorl.algorithms.td7 import TD7Agent
from evorl.envs import Box, SpaceContainer
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from tests.helpers.agents import build_agent
from tests.helpers.envs import ToyEnv


FACTORIES = {
    "td3": partial(build_agent, TD3Agent),
    "td7": partial(build_agent, TD7Agent),
    "ddpg": partial(build_agent, DDPGAgent),
    "sac": partial(build_agent, SACAgent),
    "sac_discrete": partial(build_agent, SACAgent),
    "dqn": partial(build_agent, DQNAgent),
}


def make_case(name, **kwargs):
    env = ToyEnv(4, discrete=name in ("sac_discrete", "dqn"))
    factory = FACTORIES[name]
    sizes = (
        dict(q_hidden_layer_sizes=(4,))
        if name == "dqn"
        else dict(actor_hidden_layer_sizes=(4,), critic_hidden_layer_sizes=(4,))
    )
    if name == "td7":
        sizes.update(
            z_s_dim=4,
            z_sa_dim=4,
            f_layer_sizes=(4,),
            g_layer_sizes=(4,),
            state_emb_dim=4,
            state_action_emb_dim=4,
        )
    agent = factory(env.obs_space, env.action_space, **sizes, **kwargs)
    state = agent.init(env.obs_space, env.action_space, jax.random.key(10))
    obs = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
    batch = SampleBatch(
        obs=obs,
        actions=jnp.arange(4, dtype=jnp.int32) % 2
        if env.discrete
        else jnp.full((4, 1), 0.2),
        rewards=jnp.array([1.0, -0.3, 0.5, 0.1]),
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=obs + 0.1, termination=jnp.array([0.0, 1.0, 0.0, 1.0])
            )
        ),
    )
    return agent, state, batch


@pytest.mark.parametrize("name", FACTORIES)
def test_nested_initialization_and_evaluation_keep_agent_static(name):
    agent, state, batch = make_case(name)
    env = ToyEnv(4, discrete=name in ("sac_discrete", "dqn"))
    static_fields = vars(agent).copy()
    keys = jax.random.split(jax.random.key(12), 6).reshape(2, 3)
    states = jax.jit(
        jax.vmap(jax.vmap(lambda key: agent.init(env.obs_space, env.action_space, key)))
    )(keys)
    actions = jax.jit(
        jax.vmap(
            jax.vmap(lambda s: agent.evaluate_actions(s, batch, jax.random.key(1))[0])
        )
    )(states)
    assert isinstance(states.model_state, nnx.GraphState)
    assert jax.tree.structure(states.model_state) == jax.tree.structure(
        state.model_state
    )
    for single, population in zip(
        jax.tree.leaves(state.model_state),
        jax.tree.leaves(states.model_state),
        strict=True,
    ):
        assert population.shape == (2, 3, *single.shape)
    expected_shape = (2, 3, 4) if env.discrete else (2, 3, 4, 1)
    assert actions.shape == expected_shape
    assert np.all(np.isfinite(actions))
    assert vars(agent) == static_fields
    jax.tree.map(
        np.testing.assert_array_equal,
        agent.init(env.obs_space, env.action_space, jax.random.key(10)),
        state,
    )


def test_dqn_greedy_evaluation_uses_observation_normalization(jit):
    env = ToyEnv(4, discrete=True)
    agent = build_agent(
        DQNAgent,
        env.obs_space,
        env.action_space,
        q_hidden_layer_sizes=(),
        normalize_obs=True,
    )
    state = agent.init(env.obs_space, env.action_space, jax.random.key(0))
    models = nnx.merge(state.model_state, copy=True)
    model = models.q
    model.mlp.layers[0].kernel[...] = jnp.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])
    model.mlp.layers[0].bias[...] = jnp.zeros(2)
    state = state.replace(
        model_state=nnx.unpack(models),
        obs_preprocessor_state=state.obs_preprocessor_state.replace(
            mean=jnp.full(3, 2.0)
        ),
    )
    batch = SampleBatch(obs=jnp.ones((4, 3)))
    # Raw positive observations choose action 0; normalized negative inputs
    # choose action 1, consistently with training and exploration.
    assert np.all(np.asarray(model(batch.obs).argmax(-1)) == 0)
    evaluate = jax.jit(agent.evaluate_actions) if jit else agent.evaluate_actions
    actions, _ = evaluate(state, batch, jax.random.key(1))
    np.testing.assert_array_equal(actions, np.ones(4, dtype=np.int32))


@pytest.mark.parametrize("name", FACTORIES)
def test_agent_initializes_networks_from_supplied_yaml_configuration(name):
    """Network configuration stays static; initialized variables live in GraphState."""
    root = Path(__file__).resolve().parents[2]
    agent_name = "sac" if name == "sac_discrete" else name
    config = OmegaConf.load(root / f"configs/agent/{agent_name}.yaml").agent_network
    if name == "dqn":
        config.q_hidden_layer_sizes = [7]
    else:
        config.actor_hidden_layer_sizes = [7]
        config.critic_hidden_layer_sizes = [5]
    if name == "td7":
        config.zs_dim = 4
        config.zsa_dim = 4
        config.f_layer_sizes = [4]
        config.g_layer_sizes = [4]
        config.state_emb_dim = 4
        config.state_action_emb_dim = 4
    env = ToyEnv(4, discrete=name in ("sac_discrete", "dqn"))
    factory = FACTORIES[name]
    agent = factory(env.obs_space, env.action_space, agent_network_cfg=config)
    assert agent.agent_network_cfg is config
    assert not any(name.endswith(("_network", "_graphdef")) for name in vars(agent))
    assert not any(isinstance(value, nnx.Module) for value in vars(agent).values())
    state = jax.jit(lambda key: agent.init(env.obs_space, env.action_space, key))(
        jax.random.key(24)
    )
    assert isinstance(state.model_state, nnx.GraphState)
    models = nnx.merge(state.model_state, copy=True)
    model = models.q if name == "dqn" else models.actor
    mlp = model.actor_mlp if name == "td7" else model.mlp
    assert mlp.layers[0].kernel.shape[-1] == 7
    assert state.extra_state is None or not any(
        key.endswith("_state") for key in state.extra_state
    )


@pytest.mark.parametrize("name", ["td3", "ddpg", "sac", "sac_discrete", "dqn"])
def test_network_configuration_selects_dict_observation_spaces(name, jit):
    """Policy and value networks can use differently sized observation entries."""
    obs_space = SpaceContainer(
        spaces=PyTreeDict(
            policy=Box(low=-jnp.ones(3), high=jnp.ones(3)),
            value=Box(low=-jnp.ones(5), high=jnp.ones(5)),
        )
    )
    env = ToyEnv(4, discrete=name in ("sac_discrete", "dqn"))
    factory = FACTORIES[name]
    kwargs = (
        dict(q_hidden_layer_sizes=(4,), value_obs_key="value")
        if name == "dqn"
        else dict(
            actor_hidden_layer_sizes=(4,),
            critic_hidden_layer_sizes=(4,),
            policy_obs_key="policy",
            value_obs_key="value",
        )
    )
    agent = factory(obs_space, env.action_space, **kwargs)
    initialize = lambda key: agent.init(obs_space, env.action_space, key)
    state = (jax.jit(initialize) if jit else initialize)(jax.random.key(25))
    batch = SampleBatch(obs=PyTreeDict(policy=jnp.ones((4, 3)), value=jnp.ones((4, 5))))
    evaluate = jax.jit(agent.evaluate_actions) if jit else agent.evaluate_actions
    actions, _ = evaluate(state, batch, jax.random.key(26))
    expected_shape = (4,) if env.discrete else (4, 1)
    assert actions.shape == expected_shape
    models = nnx.merge(state.model_state, copy=True)
    if name == "dqn":
        values = models.q(batch.obs)
    elif env.discrete:
        values = models.critic(batch.obs)
    else:
        values = models.critic(batch.obs, actions)
    assert values.shape[0] == 4
    assert np.isfinite(values).all()


@pytest.mark.parametrize("name", ["td3", "sac", "sac_discrete"])
def test_agent_init_validates_twin_critic_configuration(name):
    """Direct Agent construction retains the removed factories' validation."""
    env = ToyEnv(2, discrete=name == "sac_discrete")
    factory = FACTORIES[name]
    agent = factory(env.obs_space, env.action_space, num_critics=1)
    with pytest.raises(ValueError, match="num_critics must be at least 2"):
        agent.init(env.obs_space, env.action_space, jax.random.key(0))


@pytest.mark.parametrize("nested", [False, True])
def test_loss_reads_agent_state_and_preserves_nnx_spectral_updates(nested, jit):
    """The state-based loss shares only the isolated training models' variables."""
    agent, initial, batch = make_case(
        "td3", norm_layer_type="spectral_norm", normalize_obs=True
    )
    key = jax.random.key(31)
    originals = [np.array(x) for x in jax.tree.leaves(initial)]

    def differentiate(state, use_agent):
        models = agent.get_models(state, copy=True)

        def loss_fn(models):
            if use_agent:
                working_state = state.replace(model_state=nnx.unpack(models))
                losses = agent.actor_loss(working_state, batch, key)
                assert isinstance(losses, PyTreeDict)
                return losses.actor_loss
            obs = agent.obs_preprocessor(batch.obs, state.obs_preprocessor_state)
            return -models.critic(obs, models.actor(obs))[..., 0].mean()

        value, grads = nnx.value_and_grad(
            loss_fn,
            argnums=nnx.DiffState(0, nnx.All(nnx.Param, nnx.PathContains("actor"))),
        )(models)
        assert set(grads) == {"actor"}
        return value, grads, nnx.unpack(models)

    def compare(state):
        return differentiate(state, True), differentiate(state, False)

    states = initial
    fn = compare
    if nested:
        env = ToyEnv(4)
        keys = jax.random.split(jax.random.key(32), 4).reshape(2, 2)
        states = jax.vmap(
            jax.vmap(lambda k: agent.init(env.obs_space, env.action_space, k))
        )(keys)
        fn = jax.vmap(jax.vmap(compare))
    actual, expected = (jax.jit(fn) if jit else fn)(states)
    for got, want in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    before = agent.get_models(states, copy=True)
    after = nnx.merge(actual[2], copy=True)
    before_aux = jax.tree.leaves(nnx.state(before.actor, nnx.Not(nnx.Param)))
    after_aux = jax.tree.leaves(nnx.state(after.actor, nnx.Not(nnx.Param)))
    assert before_aux
    assert any(
        not np.array_equal(a, b) for a, b in zip(before_aux, after_aux, strict=True)
    )
    for original, unchanged in zip(originals, jax.tree.leaves(initial), strict=True):
        np.testing.assert_array_equal(original, unchanged)
