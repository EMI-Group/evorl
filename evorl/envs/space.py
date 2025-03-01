import chex
import jax
import jax.numpy as jnp

from evorl.types import PyTreeData


class Space(PyTreeData):
    """Base class for Space like `gym.Space`."""

    @property
    def shape(self) -> chex.Shape:
        """Get the shape of the space."""
        raise NotImplementedError

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        """Randomly sample a data in this space.

        Returns:
            A sample from the space
        """
        raise NotImplementedError

    def contains(self, x: chex.Array) -> bool:
        """Determine whether the input is in the space.

        Returns:
            A boolean value about whether x is in the space.
        """
        raise NotImplementedError


class Box(Space):
    """Continuous space in R^n.

    Attributes:
        low: The lower bounds of the box.
        high: The upper bounds of the box.
    """

    low: chex.Array
    high: chex.Array

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.uniform(
            key,
            shape=self.low.shape,
            dtype=self.low.dtype,
            minval=self.low,
            maxval=self.high,
        )

    @property
    def shape(self) -> chex.Shape:
        return self.low.shape

    def contains(self, x: chex.Array) -> chex.Array:
        return jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))


class Discrete(Space):
    """Discrete space in {0, 1, ..., n-1}.

    Attributes:
        n: The number of discrete values.
    """

    n: int

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.randint(key, shape=(), minval=0, maxval=self.n)

    @property
    def shape(self) -> chex.Shape:
        return ()

    def contains(self, x: chex.Array) -> chex.Array:
        return jnp.logical_and(x >= 0, x < self.n)
