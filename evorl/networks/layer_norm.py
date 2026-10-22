"""Normalization layers for NNX networks."""

import jax
from flax import nnx


class StaticLayerNorm(nnx.LayerNorm):
    """Layer normalization with fixed, non-trainable scale and bias.

    Args:
        num_features: Size of the normalized feature dimension.
        fixed_bias: Constant offset applied after normalization.
        fixed_scale: Constant multiplier applied after normalization.
        rngs: Random streams accepted by the NNX layer constructor.
        **kwargs: Additional arguments to `nnx.LayerNorm`.
    """

    def __init__(
        self,
        num_features: int,
        *,
        fixed_bias: float = 0.0,
        fixed_scale: float = 1.0,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        super().__init__(
            num_features, use_bias=False, use_scale=False, rngs=rngs, **kwargs
        )
        self.fixed_bias = fixed_bias
        self.fixed_scale = fixed_scale

    def __call__(self, x: jax.Array, *, mask: jax.Array | None = None):
        return super().__call__(x, mask=mask) * self.fixed_scale + self.fixed_bias


def get_norm_layer(norm_layer_type: str) -> type[nnx.Module] | None:
    """Return the normalization layer class for an MLP hidden layer."""
    match norm_layer_type:
        case "layer_norm":
            return nnx.LayerNorm
        case "static_layer_norm":
            return StaticLayerNorm
        case "none":
            return None
        case _:
            raise ValueError(f"Invalid norm_layer_type: {norm_layer_type}")
