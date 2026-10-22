"""Atari NNX encoders preserve image preprocessing and batch axes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from evorl.networks.cnn.atari_cnn import CNN_AgentStem, CNN_QNetwork, make_cnn_agent


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_atari_q_preprocessing_matches_normalized_image_input(batch_shape):
    raw = CNN_QNetwork((4, 40, 40), 3, normalize=True, rngs=nnx.Rngs(0))
    normalized = CNN_QNetwork((40, 40, 4), 3, normalize=False, rngs=nnx.Rngs(0))
    x = jax.random.randint(
        jax.random.key(1), (*batch_shape, 4, 40, 40), 0, 256, dtype=jnp.uint8
    )
    expected = normalized(jnp.moveaxis(x, -3, -1).astype(jnp.float32) / 255)
    np.testing.assert_allclose(raw(x), expected, rtol=1e-6, atol=1e-6)
    assert expected.shape == (*batch_shape, 3)


def test_atari_shared_stem_heads_and_functional_jit():
    stem, actor, critic = make_cnn_agent(
        (4, 40, 40), 3, hidden_size=8, rngs=nnx.Rngs(0)
    )
    model = nnx.Dict(stem=stem, actor=actor, critic=critic)
    graphdef, state = nnx.split(model)

    @jax.jit
    def forward(state, obs):
        local = nnx.merge(graphdef, state, copy=True)
        features = local.stem(obs)
        return local.actor(features), local.critic(features)

    policy, value = forward(state, jnp.zeros((2, 4, 40, 40), dtype=jnp.uint8))
    assert policy.shape == (2, 3)
    assert value.shape == (2, 1)
    normalized = CNN_AgentStem(
        (40, 40, 4), last_hidden_size=8, normalize=True, rngs=nnx.Rngs(0)
    )
    np.testing.assert_allclose(
        stem(jnp.full((2, 4, 40, 40), 255, dtype=jnp.uint8)),
        normalized(jnp.ones((2, 40, 40, 4))),
        rtol=1e-6,
    )
