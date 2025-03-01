import jax.numpy as jnp


def sort_by_key(keys, *vals):
    assert len(keys.shape) == 1, (
        f"Expect keys to be a 1d-vector, got shape {keys.shape}."
    )
    order = jnp.argsort(keys)
    vals = map(lambda v: v[order], vals)
    return keys[order], *vals
