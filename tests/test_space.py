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

    data = space.sample(jax.random.PRNGKey(0))

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

    data = space.sample(jax.random.PRNGKey(0))

    data_shape = space.shape
    assert isinstance(data_shape, dict)
    assert data_shape["c"] == space.spaces["c"].shape

    new_data = dict(data)
    new_data["c"] = dict(new_data["c"])
    new_data["c"]["a1"] = new_data["c"]["a1"] + 10

    assert not space.contains(new_data)
