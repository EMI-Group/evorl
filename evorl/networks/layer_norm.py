from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


class StaticLayerNorm(nn.LayerNorm):
    """Layer normalization with fixed scale and bias."""

    use_bias: bool = False
    use_scale: bool = False
    fixed_bias: jax.Array = jnp.zeros(())
    fixed_scale: jax.Array = jnp.ones(())

    @nn.compact
    def __call__(self, x, *, mask: Optional[jax.Array] = None):
        y = super().__call__(x, mask=mask)
        return y * self.fixed_scale + self.fixed_bias


def get_norm_layer(norm_layer_type: str) -> type[nn.Module]:
    """Get the normalization layer class based on the type."""
    match norm_layer_type:
        case "layer_norm":
            norm_layer = nn.LayerNorm
        case "static_layer_norm":
            norm_layer = StaticLayerNorm
        case "none":
            norm_layer = None
        case _:
            raise ValueError(f"Invalid norm_layer_type: {norm_layer_type}")

    return norm_layer
