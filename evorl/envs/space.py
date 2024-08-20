import chex
import jax
import jax.numpy as jnp

from evorl.types import PyTreeData


class Space(PyTreeData):
    """
    a jax version of the gym.Space
    """

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        """Randomly sample an element of this space.

        Can be uniform or non-uniform sampling based on boundedness of space.

        Returns:
            A sampled actions from the space
        """
        raise NotImplementedError

    @property
    def shape(self) -> chex.Shape:
        """Return the shape of the space"""
        raise NotImplementedError

    def contains(self, x: chex.Array) -> bool:
        """Return True if x is a valid member of the space."""
        raise NotImplementedError


class Box(Space):
    low: chex.Array
    high: chex.Array

    # def __post_init__(self):
    #     chex.assert_trees_all_equal_dtypes(
    #         self.low,
    #         self.high,
    #         custom_message="low and high should have the same shape and dtype!",
    #     )
    #     assert (
    #         self.high >= self.low
    #     ).all(), "high should be greater than or equal to low"

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
    n: int

    # def __post_init__(self):
    #     assert self.n > 0, "n should be a positive integer"

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.randint(key, shape=(), minval=0, maxval=self.n)

    @property
    def shape(self) -> chex.Shape:
        return ()

    def contains(self, x: chex.Array) -> chex.Array:
        return jnp.logical_and(x >= 0, x < self.n)
