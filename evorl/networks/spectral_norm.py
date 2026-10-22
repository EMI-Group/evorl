# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NNX dense layers with spectral normalization.

Adapted from Brax's training networks. The singular vector is model state rather
than a trainable parameter; gradient updates only optimize the original kernel.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax

_LECUN_NORMAL = jax.nn.initializers.lecun_normal()


class SingularVector(nnx.Variable):
    """Power-iteration state, excluded from parameter gradients and EC operators."""


def _l2_normalize(x, axis=None, eps=1e-12):
    return x * lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


class SNDense(nnx.Linear):
    """Linear layer with power-iteration spectral normalization.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Whether to add a trainable bias.
        dtype: Computation dtype.
        precision: Precision of the matrix multiplication.
        kernel_init: Initializer of the unnormalized trainable kernel.
        bias_init: Initializer of the trainable bias.
        eps: Stabilizer for power-iteration norms and the singular-value divisor.
        n_steps: Number of power iterations per forward call.
        rngs: Random streams for parameters and the initial singular vector.
            A `sing_vec` stream can be provided independently of `params`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        dtype: Any = jnp.float32,
        precision: Any = None,
        kernel_init: Callable = _LECUN_NORMAL,
        bias_init: Callable = jax.nn.initializers.zeros,
        eps: float = 1e-4,
        n_steps: int = 1,
        rngs: nnx.Rngs,
    ):
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        super().__init__(
            in_features,
            out_features,
            use_bias=use_bias,
            dtype=dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )
        self.eps = eps
        self.n_steps = n_steps
        self.u0 = SingularVector(
            jax.random.normal(rngs.sing_vec(), (1, out_features), dtype=jnp.float32)
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Apply the normalized kernel and advance the singular-vector state."""
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = jnp.asarray(self.kernel[...], self.dtype)
        u0 = self.u0[...]
        for _ in range(self.n_steps):
            v0 = _l2_normalize(u0 @ kernel.T, eps=self.eps)
            u0 = _l2_normalize(v0 @ kernel, eps=self.eps)
        u0 = lax.stop_gradient(u0)
        v0 = lax.stop_gradient(v0)
        sigma = jnp.maximum((v0 @ kernel @ u0.T)[0, 0], self.eps)
        # A zero kernel has no singular direction. Keep the previous estimate so
        # power iteration can recover after an optimizer makes the kernel nonzero.
        self.u0[...] = jnp.where(jnp.any(u0 != 0), u0, self.u0[...])
        output = lax.dot_general(
            inputs,
            kernel / sigma,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if self.bias is not None:
            output = output + jnp.asarray(self.bias[...], self.dtype)
        return output
