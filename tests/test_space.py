import jax
import jax.numpy as jnp
from evorl.envs import Box, Discrete, SpaceContainer


def test_container():
    shape = (7, 3)
    high = jnp.ones(shape)

    space = SpaceContainer(
        spaces={
            "a": Box(low=-high, high=high),
            "b": Discrete(n=5),
            "c": Box(low=-2 * high, high=5 * high),
        }
    )

    data = space.sample(jax.random.key(0))

    data_shape = space.shape
    assert isinstance(data_shape, dict)
    assert (
        data_shape["a"] == shape and data_shape["b"] == () and data_shape["c"] == shape
    )

    assert (
        data["a"].shape == shape and data["b"].shape == () and data["c"].shape == shape
    )

    new_data = dict(data)

    new_data["a"] = new_data["a"] + 10

    assert not space.contains(new_data)


def test_nested_container():
    shape = (7, 3)
    high = jnp.ones(shape)

    space = SpaceContainer(
        spaces={
            "a": Box(low=-high, high=high),
            "b": Discrete(n=5),
            "c": SpaceContainer(
                spaces={
                    "a1": Box(low=-2 * high, high=2 * high),
                    "b1": Discrete(n=13),
                }
            ),
        }
    )

    data = space.sample(jax.random.key(0))

    data_shape = space.shape
    assert isinstance(data_shape, dict)
    assert data_shape["c"] == space.spaces["c"].shape

    new_data = dict(data)
    new_data["c"] = dict(new_data["c"])
    new_data["c"]["a1"] = new_data["c"]["a1"] + 10

    assert not space.contains(new_data)


def test_multidiscrete_preserves_array_shape_and_integer_bounds():
    from evorl.envs import MultiDiscrete

    space = MultiDiscrete(nvec=jnp.array([[2, 3], [1, 5]], dtype=jnp.int16))
    samples = jax.jit(jax.vmap(space.sample))(jax.random.split(jax.random.key(0), 64))
    assert samples.shape == (64, 2, 2)
    assert samples.dtype == jnp.int16
    assert jax.vmap(space.contains)(samples).all()
    assert not space.contains(jnp.array([[2, 0], [0, 0]]))
    assert not space.contains(jnp.zeros((2, 2), dtype=jnp.float32))
    assert not space.contains(jnp.zeros(4, dtype=jnp.int32))


def test_box_samples_integer_and_boolean_observations():
    import numpy as np

    for dtype in [jnp.int8, jnp.uint8, jnp.int16, jnp.int32]:
        low = jnp.array([0, 0, 5, 125], dtype=dtype)
        high = jnp.array([1, 0, 10, 127], dtype=dtype)
        space = Box(low=low, high=high)
        samples = jax.jit(jax.vmap(space.sample))(
            jax.random.split(jax.random.key(0), 128)
        )
        assert samples.dtype == dtype
        assert jax.vmap(space.contains)(samples).all()
        # Integer upper bounds are inclusive; int8 arithmetic must not overflow.
        np.testing.assert_array_equal(jnp.unique(samples[:, 0]), [0, 1])
        np.testing.assert_array_equal(jnp.unique(samples[:, 3]), [125, 126, 127])
    space = Box(
        low=jnp.array([0, 255], dtype=jnp.uint8),
        high=jnp.array([255, 255], dtype=jnp.uint8),
    )
    samples = jax.jit(jax.vmap(space.sample))(jax.random.split(jax.random.key(0), 64))
    assert samples.dtype == jnp.uint8
    assert jax.vmap(space.contains)(samples).all()
    np.testing.assert_array_equal(samples[:, 1], 255)
    space = Box(low=jnp.array([False, True]), high=jnp.array([True, True]))
    samples = jax.jit(jax.vmap(space.sample))(jax.random.split(jax.random.key(0), 64))
    assert samples.dtype == jnp.bool_
    assert jax.vmap(space.contains)(samples).all()
    np.testing.assert_array_equal(jnp.unique(samples[:, 0]), [False, True])
