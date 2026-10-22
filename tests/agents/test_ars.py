"""Zero-initialized ARS policies retain native parameter and variable types."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from evorl.algorithms.contrib.ars_linear import ARSAgent
from omegaconf import OmegaConf
from evorl.envs import Box
from evorl.sample_batch import SampleBatch


@pytest.mark.parametrize(
    "norm", ["none", "layer_norm", "static_layer_norm", "spectral_norm"]
)
def test_ars_zero_initialized_policy_and_independent_states(norm, jit):
    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = Box(low=-jnp.ones(2), high=jnp.ones(2))
    agent = ARSAgent(
        agent_network_cfg=OmegaConf.create(
            dict(
                actor_hidden_layer_sizes=[4],
                norm_layer_type=norm,
                use_bias=True,
            )
        )
    )
    initialize = jax.jit(agent.init) if jit else agent.init
    state = initialize(obs_space, action_space, jax.random.key(0))
    evaluate = jax.jit(agent.evaluate_actions) if jit else agent.evaluate_actions
    actions, _ = evaluate(state, SampleBatch(obs=jnp.ones((5, 3))), jax.random.key(1))
    np.testing.assert_array_equal(actions, jnp.zeros((5, 2)))
    assert all(
        isinstance(x, nnx.Param)
        for _, x in nnx.to_flat_state(
            nnx.state(nnx.merge(state.model_state, copy=True).policy, nnx.Param)
        )
    )
    if norm == "spectral_norm":
        assert jax.tree.leaves(
            nnx.state(
                nnx.merge(state.model_state, copy=True).policy, nnx.Not(nnx.Param)
            )
        )

    # A nonzero output bias exposes clipping in both action paths. Zero kernels
    # keep the expected actions independent of normalization and input values.
    models = agent.get_models(state, copy=True)
    models.policy.layers[-1].bias[...] = jnp.array([-2.0, 3.0])
    clipped_state = state.replace(model_state=nnx.unpack(models))
    for action_fn in (agent.compute_actions, agent.evaluate_actions):
        action_fn = jax.jit(action_fn) if jit else action_fn
        clipped, _ = action_fn(
            clipped_state, SampleBatch(obs=jnp.ones((5, 3))), jax.random.key(2)
        )
        np.testing.assert_array_equal(clipped, jnp.tile(jnp.array([-1.0, 1.0]), (5, 1)))
