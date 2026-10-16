"""Regression checks shared by the minimum and latest JAX environments."""

import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from evorl.algorithms.td7 import avg_l1_norm
from evorl.distributed import get_process_id, is_dist_initialized
from evorl.networks.linear import MLP


def test_linen_functional_training_step():
    model = MLP(layer_sizes=(8, 2))
    inputs = jnp.ones((4, 3))
    params = model.init(jax.random.key(0), inputs)
    optimizer = optax.adam(1e-2)

    def loss(params):
        return jnp.mean(model.apply(params, inputs) ** 2)

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


def test_shard_map_wrappers_and_gradients_on_two_cpu_devices():
    code = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from evorl.distributed import all_gather, shmap_map, shmap_vmap

        assert jax.device_count() == 2
        mesh = jax.make_mesh((len(jax.devices()),), ("devices",), axis_types=(jax.sharding.AxisType.Explicit,), devices=jax.devices())
        jax.set_mesh(mesh)
        inputs = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
        inputs = jax.device_put(inputs, jax.P("devices"))
        # Four logical random streams on two devices: each shard has two keys.
        keys = jax.device_put(jax.random.split(jax.random.key(7), 4), jax.P("devices"))
        assert keys.shape == (4,)
        assert keys.addressable_shards[0].data.shape == (2,)
        assert jax.dtypes.issubdtype(keys.dtype, jax.dtypes.prng_key)
        sample = jax.jit(shmap_vmap(
            lambda key: jax.random.uniform(key),
            in_specs=jax.P("devices"), out_specs=jax.P("devices"),
        ))
        np.testing.assert_array_equal(
            sample(keys), jax.vmap(jax.random.uniform)(jax.random.split(jax.random.key(7), 4))
        )


        for wrapper in (shmap_map, shmap_vmap):
            mapped = jax.jit(wrapper(
                lambda row: row ** 2,
                in_specs=jax.P("devices"), out_specs=jax.P("devices"),
            ))
            np.testing.assert_allclose(mapped(inputs), inputs ** 2)
            grads = jax.jit(jax.grad(lambda x: jnp.sum(mapped(x))))(inputs)
            np.testing.assert_allclose(grads, 2 * inputs)

        reduced = jax.jit(jax.shard_map(
            lambda block: jax.lax.pmean(block.sum(axis=0), "devices"),
            in_specs=jax.P("devices"), out_specs=jax.P(),
        ))
        np.testing.assert_allclose(reduced(inputs), inputs.sum(axis=0) / 2)
        grads = jax.jit(jax.grad(lambda x: jnp.sum(reduced(x))))(inputs)
        np.testing.assert_allclose(grads, jnp.full_like(inputs, 0.5))

        gathered = jax.jit(jax.shard_map(
            lambda block: all_gather(
                block, "devices", axis=0, tiled=True, to="invarying"
            ),
            in_specs=jax.P("devices"), out_specs=jax.P(),
        ))
        np.testing.assert_array_equal(gathered(inputs), inputs)
        grads = jax.jit(jax.grad(lambda x: jnp.sum(gathered(x) ** 2)))(inputs)
        np.testing.assert_allclose(grads, 2 * inputs)
        hlo = gathered.lower(inputs).compile().as_text().lower()
        assert "all-gather(" in hlo
        assert "all-reduce(" not in hlo
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            "JAX_NUM_CPU_DEVICES": "2",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "norm", ["none", "layer_norm", "static_layer_norm", "spectral_norm"]
)
@pytest.mark.parametrize("vmapped", [False, True])
def test_linen_normalization_state_and_gradients(norm, vmapped):
    from evorl.networks.linear import make_mlp, make_vmap_mlp

    factory = make_vmap_mlp if vmapped else make_mlp
    model = factory((8, 2), norm_layer_type=norm)
    inputs = jnp.ones((3, 4, 5) if vmapped else (4, 5))
    params_key, state_key = jax.random.split(jax.random.key(0))
    variables = model.init({"params": params_key, "sing_vec": state_key}, inputs)

    @jax.jit
    def step(variables):
        def loss(params):
            output, new_state = model.apply(
                {**variables, "params": params},
                inputs,
                rngs={"sing_vec": state_key},
                mutable=["sing_vec"],
            )
            return jnp.mean(output**2), (output, new_state)

        (value, (output, new_state)), grads = jax.value_and_grad(loss, has_aux=True)(
            variables["params"]
        )
        return value, output, new_state, grads

    value, output, new_state, grads = step(variables)
    assert jnp.isfinite(value)
    assert output.shape == ((4, 3, 2) if vmapped else (4, 2))
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads))
    if norm == "spectral_norm":
        assert "sing_vec" in new_state
        if vmapped:
            assert all(x.shape[0] == 3 for x in jax.tree.leaves(new_state["sing_vec"]))


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
    from evorl.algorithms.a2c import make_mlp_a2c_agent
    from evorl.algorithms.dqn import make_mlp_discrete_dqn_agent
    from evorl.algorithms.ppo import make_mlp_ppo_agent
    from evorl.algorithms.sac import make_mlp_sac_agent
    from evorl.algorithms.td3 import make_mlp_td3_agent
    from evorl.algorithms.td7 import make_td7_agent
    from evorl.envs import Box, Discrete
    from evorl.sample_batch import SampleBatch

    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = (
        Discrete(n=3)
        if name in ("dqn", "sac_discrete")
        else Box(low=-jnp.ones(2), high=jnp.ones(2))
    )
    sizes = dict(actor_hidden_layer_sizes=(8,), critic_hidden_layer_sizes=(8,))
    factories = dict(
        a2c=make_mlp_a2c_agent,
        ppo=make_mlp_ppo_agent,
        sac=make_mlp_sac_agent,
        sac_discrete=make_mlp_sac_agent,
        td3=make_mlp_td3_agent,
    )
    if name == "dqn":
        agent = make_mlp_discrete_dqn_agent(
            action_space, q_hidden_layer_sizes=(8,), normalize_obs=True
        )
    elif name == "td7":
        agent = make_td7_agent(
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
        agent = factories[name](action_space, **sizes, normalize_obs=True)
    init_key, action_key = jax.random.split(jax.random.key(0))
    state = jax.jit(agent.init)(obs_space, action_space, init_key)
    batch = SampleBatch(obs=jnp.ones((4, 3)))
    actions, _ = jax.jit(agent.compute_actions)(state, batch, action_key)
    evaluated, _ = jax.jit(agent.evaluate_actions)(state, batch, action_key)
    assert actions.shape == (4, *action_space.shape)
    assert evaluated.shape == actions.shape
    assert jnp.all(jnp.isfinite(actions))
