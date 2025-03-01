import math
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class CNN_QNetwork(nn.Module):
    """A simple CNN network for Q-learning on Atari Games."""

    action_dim: int
    normalize: bool = False  # for native Atari envs

    @nn.compact
    def __call__(self, x):
        if self.normalize:
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = x / (255.0)
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class CNN_AgentStem(nn.Module):
    """A simple CNN backbone used for the agent and the critic."""

    last_hidden_size: int = 512
    normalize: bool = False

    @nn.compact
    def __call__(self, x):
        if not self.normalize:
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(math.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(math.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(math.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        # flatten to 1d
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            self.last_hidden_size,
            kernel_init=orthogonal(math.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    """Lineat Critic Head."""

    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    """Linear Actor Head."""

    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)


def make_cnn_agent(
    obs_shape: tuple[int],
    action_size: int,
    hidden_size: int = 512,
):
    stem = CNN_AgentStem(last_hidden_size=hidden_size)
    actor_header = Actor(action_dim=action_size)
    critic_header = Critic()

    dummy_obs = jnp.zeros((1, *obs_shape))
    # dummy_action = jnp.zeros((1, action_size))

    def agent_init_fn(rng):
        stem_key, actor_key, critic_key = jax.random.split(rng, num=3)
        dummy_hidden, stem_params = stem.init_with_output(stem_key, dummy_obs)
        actor_params = actor_header.init(actor_key, dummy_hidden)
        critic_params = critic_header.init(critic_key, dummy_hidden)

        return stem_params, actor_params, critic_params

    return (stem, actor_header, critic_header), agent_init_fn
