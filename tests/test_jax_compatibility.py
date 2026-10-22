"""Regression checks shared by the minimum and latest JAX environments."""

from functools import partial
from tests.helpers.agents import build_agent

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from evorl.algorithms.td7 import avg_l1_norm
from evorl.distributed import get_process_id, is_dist_initialized
from evorl.networks.linear import MLP


def test_nnx_functional_training_step():
    model = MLP(3, (8, 2), rngs=nnx.Rngs(0))
    inputs = jnp.ones((4, 3))
    graphdef, params = nnx.split(model)
    optimizer = optax.adam(1e-2)

    def loss(params):
        return jnp.mean(nnx.merge(graphdef, params, copy=True)(inputs) ** 2)

    @jax.jit
    def step(params, opt_state):
        grads = jax.grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    updated, _ = step(params, optimizer.init(params))
    assert loss(updated) < loss(params)


def test_td7_normalization_handles_zero_and_nonzero_inputs():
    inputs = jnp.array([[0.0, 0.0], [1.0, -3.0]])
    normalized = jax.jit(avg_l1_norm)(inputs)
    np.testing.assert_allclose(normalized, [[0.0, 0.0], [0.5, -1.5]])
    grads = jax.grad(lambda x: jnp.sum(avg_l1_norm(x)))(inputs)
    assert jnp.all(jnp.isfinite(grads))


def test_single_process_distributed_status():
    assert not is_dist_initialized()
    assert get_process_id() == 0


@pytest.mark.parametrize(
    "norm", ["none", "layer_norm", "static_layer_norm", "spectral_norm"]
)
@pytest.mark.parametrize("vmapped", [False, True])
def test_nnx_normalization_state_and_gradients(norm, vmapped):
    from evorl.networks.linear import make_mlp, make_vmap_mlp
    from evorl.networks.spectral_norm import SingularVector

    model = (
        make_vmap_mlp(5, (8, 2), 3, norm_layer_type=norm, rngs=nnx.Rngs(0))
        if vmapped
        else make_mlp(5, (8, 2), norm_layer_type=norm, rngs=nnx.Rngs(0))
    )
    inputs = jnp.ones((3, 4, 5) if vmapped else (4, 5))
    graphdef, params, other = nnx.split(model, nnx.Param, ...)

    @jax.jit
    def step(params, other):
        def loss(params):
            local = nnx.merge(graphdef, params, other, copy=True)
            output = local(inputs)
            return jnp.mean(output**2), (output, nnx.state(local, SingularVector))

        return jax.value_and_grad(loss, has_aux=True)(params)

    (value, (output, new_state)), grads = step(params, other)
    assert jnp.isfinite(value)
    assert output.shape == ((4, 3, 2) if vmapped else (4, 2))
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads))
    if norm == "spectral_norm":
        assert jax.tree.leaves(new_state)
        if vmapped:
            assert all(x.shape[0] == 3 for x in jax.tree.leaves(new_state))


@pytest.mark.parametrize("optimizer_name", ["adam", "sgd", "rmsprop"])
def test_openes_optax_injected_hyperparameters(optimizer_name):
    from evorl.ec.optimizers import ExponentialScheduleSpec, OpenES

    es = OpenES(
        pop_size=4,
        lr_schedule=ExponentialScheduleSpec(init=0.1, final=0.01, decay=0.5),
        noise_std_schedule=ExponentialScheduleSpec(init=0.2, final=0.02, decay=0.5),
        optimizer_name=optimizer_name,
    )
    state = es.init({"w": jnp.ones(3)}, jax.random.key(0))
    for _ in range(2):
        population, state = jax.jit(es.ask)(state)
        fitness = -jnp.sum(population["w"] ** 2, axis=-1)
        _, state = jax.jit(es.tell)(state, fitness)
    np.testing.assert_allclose(state.opt_state.hyperparams["learning_rate"], 0.0325)
    np.testing.assert_allclose(state.noise_std, 0.065)
    assert jnp.all(jnp.isfinite(state.mean["w"]))


def test_running_statistics_shape_validation_and_numpy_api():
    from evorl.utils import running_statistics

    data = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    state = running_statistics.init_state(data[0])
    updated = jax.jit(running_statistics.update)(state, data)
    np.testing.assert_allclose(updated.mean, np.mean(np.asarray(data), axis=0))
    np.testing.assert_allclose(updated.std, np.std(np.asarray(data), axis=0))
    with pytest.raises(AssertionError):
        running_statistics.update(state, jnp.ones((4, 2)))


def test_pmap_method_axis_name():
    from evorl.utils.jax_utils import pmap_method

    @pmap_method("devices")
    def total(x):
        return jax.lax.psum(x, "devices")

    np.testing.assert_array_equal(total(jnp.ones(1)), jnp.ones(1))


@pytest.mark.parametrize(
    "name", ["dqn", "a2c", "ppo", "sac", "sac_discrete", "td3", "td7"]
)
def test_agents_initialize_and_sample_with_typed_keys(name):
    from evorl.algorithms.a2c import A2CAgent
    from evorl.algorithms.dqn import DQNAgent
    from evorl.algorithms.ppo import PPOAgent
    from evorl.algorithms.sac import SACAgent
    from evorl.algorithms.td3 import TD3Agent
    from evorl.algorithms.td7 import TD7Agent
    from evorl.envs import Box, Discrete
    from evorl.sample_batch import SampleBatch

    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = (
        Discrete(n=3)
        if name in ("dqn", "sac_discrete")
        else Box(low=-jnp.ones(2), high=jnp.ones(2))
    )
    sizes = {"actor_hidden_layer_sizes": (8,), "critic_hidden_layer_sizes": (8,)}
    factories = {
        "a2c": partial(build_agent, A2CAgent),
        "ppo": partial(build_agent, PPOAgent),
        "sac": partial(build_agent, SACAgent),
        "sac_discrete": partial(build_agent, SACAgent),
        "td3": partial(build_agent, TD3Agent),
    }
    if name == "dqn":
        agent = build_agent(
            DQNAgent,
            obs_space,
            action_space,
            q_hidden_layer_sizes=(8,),
            normalize_obs=True,
        )
    elif name == "td7":
        agent = build_agent(
            TD7Agent,
            obs_space,
            action_space,
            **sizes,
            z_s_dim=4,
            z_sa_dim=4,
            f_layer_sizes=(8,),
            g_layer_sizes=(8,),
            state_emb_dim=4,
            state_action_emb_dim=4,
            normalize_obs=True,
        )
    else:
        agent = factories[name](obs_space, action_space, **sizes, normalize_obs=True)
    init_key, action_key = jax.random.split(jax.random.key(0))
    state = jax.jit(lambda key: agent.init(obs_space, action_space, key))(init_key)
    batch = SampleBatch(obs=jnp.ones((4, 3)))
    actions, _ = jax.jit(agent.compute_actions)(state, batch, action_key)
    evaluated, _ = jax.jit(agent.evaluate_actions)(state, batch, action_key)
    assert actions.shape == (4, *action_space.shape)
    assert evaluated.shape == actions.shape
    assert jnp.all(jnp.isfinite(actions))
