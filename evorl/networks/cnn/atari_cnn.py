"""NNX convolutional networks for Atari image observations."""

import math

import jax
import jax.numpy as jnp
from flax import nnx


class CNN_QNetwork(nnx.Module):
    """Atari Q network with optional uint8 NCHW input preprocessing.

    Args:
        obs_shape: Shape of one observation, excluding batch dimensions. It is
            NCHW without N when `normalize=True`, otherwise HWC.
        action_dim: Number of discrete actions.
        normalize: Whether to convert CHW bytes to HWC floats in `[0, 1]`.
        rngs: Random streams for parameter initialization.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        normalize: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.normalize = normalize
        height, width, channels = (
            (obs_shape[1], obs_shape[2], obs_shape[0]) if normalize else obs_shape
        )
        self.conv1 = nnx.Conv(
            channels, 32, (8, 8), strides=4, padding="VALID", rngs=rngs
        )
        self.conv2 = nnx.Conv(32, 64, (4, 4), strides=2, padding="VALID", rngs=rngs)
        self.conv3 = nnx.Conv(64, 64, (3, 3), padding="VALID", rngs=rngs)
        for kernel, stride in ((8, 4), (4, 2), (3, 1)):
            height, width = (
                (height - kernel) // stride + 1,
                (width - kernel) // stride + 1,
            )
        if height <= 0 or width <= 0:
            raise ValueError("Observation is too small for the Atari convolution stack")
        self.hidden = nnx.Linear(height * width * 64, 512, rngs=rngs)
        self.output = nnx.Linear(512, action_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.normalize:
            x = jnp.moveaxis(x, -3, -1).astype(jnp.float32) / 255.0
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = x.reshape((*x.shape[:-3], -1))
        return self.output(jax.nn.relu(self.hidden(x)))


class CNN_AgentStem(nnx.Module):
    """Shared Atari encoder for policy and value heads.

    `normalize=False` preserves the original interface: inputs are CHW bytes
    requiring preprocessing. `normalize=True` accepts already normalized HWC
    observations. Arbitrary leading batch dimensions are preserved.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        last_hidden_size: int = 512,
        normalize: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.normalize = normalize
        height, width, channels = (
            obs_shape if normalize else (obs_shape[1], obs_shape[2], obs_shape[0])
        )
        kernel_init = jax.nn.initializers.orthogonal(math.sqrt(2))
        self.conv1 = nnx.Conv(
            channels,
            32,
            (8, 8),
            strides=4,
            padding="VALID",
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            32,
            64,
            (4, 4),
            strides=2,
            padding="VALID",
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            64, 64, (3, 3), padding="VALID", kernel_init=kernel_init, rngs=rngs
        )
        for kernel, stride in ((8, 4), (4, 2), (3, 1)):
            height, width = (
                (height - kernel) // stride + 1,
                (width - kernel) // stride + 1,
            )
        if height <= 0 or width <= 0:
            raise ValueError("Observation is too small for the Atari convolution stack")
        self.hidden = nnx.Linear(
            height * width * 64, last_hidden_size, kernel_init=kernel_init, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        if not self.normalize:
            x = jnp.moveaxis(x, -3, -1).astype(jnp.float32) / 255.0
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = x.reshape((*x.shape[:-3], -1))
        return jax.nn.relu(self.hidden(x))


class Critic(nnx.Linear):
    """Scalar value head on top of an encoder."""

    def __init__(self, input_size: int, *, rngs: nnx.Rngs):
        super().__init__(
            input_size, 1, kernel_init=jax.nn.initializers.orthogonal(1), rngs=rngs
        )


class Actor(nnx.Linear):
    """Discrete-action policy head on top of an encoder."""

    def __init__(self, input_size: int, action_dim: int, *, rngs: nnx.Rngs):
        super().__init__(
            input_size,
            action_dim,
            kernel_init=jax.nn.initializers.orthogonal(0.01),
            rngs=rngs,
        )


def make_cnn_agent(
    obs_shape: tuple[int, int, int],
    action_size: int,
    hidden_size: int = 512,
    *,
    rngs: nnx.Rngs,
) -> tuple[CNN_AgentStem, Actor, Critic]:
    """Construct a shared Atari stem and separately addressable policy/value heads."""
    return (
        CNN_AgentStem(obs_shape, last_hidden_size=hidden_size, rngs=rngs),
        Actor(hidden_size, action_dim=action_size, rngs=rngs),
        Critic(hidden_size, rngs=rngs),
    )
