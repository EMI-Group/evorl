import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import PyTreeData
from evorl.utils.jax_utils import rng_split_like_tree


class Space(PyTreeData):
    """Base class for Space like `gym.Space`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the space."""
        raise NotImplementedError

    def sample(self, key: jax.Array) -> jax.Array:
        """Randomly sample a data in this space.

        Returns:
            A sample from the space
        """
        raise NotImplementedError

    def contains(self, x: jax.typing.ArrayLike) -> bool:
        """Determine whether the input is in the space.

        Returns:
            A boolean value about whether x is in the space.
        """
        raise NotImplementedError


class Box(Space):
    """Array space with continuous, integer or boolean bounds.

    Attributes:
        low: The lower bounds of the box.
        high: The upper bounds of the box.
    """

    low: jax.Array
    high: jax.Array

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample within the bounds, including integer upper bounds.

        Integer bounds use at least `int32` for arithmetic. `high + 1` must
        fit that sampling dtype.
        """
        dtype = self.low.dtype
        if jnp.issubdtype(dtype, jnp.bool_):
            return jnp.clip(
                jax.random.bernoulli(key, shape=self.shape), self.low, self.high
            )
        if jnp.issubdtype(dtype, jnp.integer):
            # Widen bounds before adding one, e.g. uint8 image bounds [0, 255].
            bounds_dtype = jnp.promote_types(dtype, jnp.int32)
            return jax.random.randint(
                key,
                self.shape,
                self.low,
                self.high.astype(bounds_dtype) + 1,
                dtype=dtype,
            )
        return jax.random.uniform(
            key,
            shape=self.low.shape,
            dtype=self.low.dtype,
            minval=self.low,
            maxval=self.high,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape

    def contains(self, x: jax.typing.ArrayLike) -> jax.Array:
        return jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))


class Discrete(Space):
    """Discrete space in {0, 1, ..., n-1}.

    Attributes:
        n: The number of discrete values.
    """

    n: int

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.randint(key, shape=(), minval=0, maxval=self.n)

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def contains(self, x: jax.typing.ArrayLike) -> jax.Array:
        return jnp.logical_and(x >= 0, x < self.n)


class MultiDiscrete(Space):
    """An integer array with a separate category count for each element.

    Attributes:
        nvec: Positive category counts. Samples have the same shape and integer
            dtype, and each element lies in `[0, nvec)`.
    """

    nvec: jax.Array

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.randint(
            key, self.shape, minval=0, maxval=self.nvec, dtype=self.nvec.dtype
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.nvec.shape

    def contains(self, x: jax.typing.ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        if x.shape != self.shape or not jnp.issubdtype(x.dtype, jnp.integer):
            return jnp.asarray(False)
        return jnp.all((x >= 0) & (x < self.nvec))


class SpaceContainer(Space):
    """Container for structural spaces.

    Attributes:
        spaces: a pytree of spaces.
    """

    spaces: chex.ArrayTree

    def sample(self, key: jax.Array) -> chex.ArrayTree:
        keys = rng_split_like_tree(
            key, self.spaces, is_leaf=lambda s: isinstance(s, Space)
        )
        return jtu.tree_map(
            lambda s, key: s.sample(key),
            self.spaces,
            keys,
            is_leaf=lambda s: isinstance(s, Space),
        )

    @property
    def shape(self) -> chex.ArrayTree:
        return jtu.tree_map(
            lambda s: s.shape,
            self.spaces,
            is_leaf=lambda s: isinstance(s, Space),
        )

    def contains(self, data: chex.ArrayTree) -> chex.ArrayTree:
        return jtu.tree_all(
            jtu.tree_map(
                lambda s, x: s.contains(x),
                self.spaces,
                data,
                is_leaf=lambda s: isinstance(s, Space),
            )
        )


def is_leaf_space(space):
    return isinstance(space, Space) and not isinstance(space, SpaceContainer)
