# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Network definitions."""


from typing import Any, Callable, Sequence, Tuple, Optional
import warnings

from brax.training import types
from .spectral_norm import SNDense
from flax import linen as nn
import jax
import jax.numpy as jnp
from functools import partial

ActivationFn = Callable[[jax.Array], jax.Array]
Initializer = Callable[..., Any]


class MLP(nn.Module):
    """MLP module."""
    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activation_final: Optional[ActivationFn] = None
    bias: bool = True
    norm_layer: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, data: jax.Array):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = nn.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                use_bias=self.bias)(hidden)

            if i != len(self.layer_sizes) - 1:
                if self.norm_layer is not None:
                    hidden = self.norm_layer()(hidden)

                hidden = self.activation(hidden)
            elif self.activation_final is not None:
                if self.norm_layer is not None:
                    hidden = self.norm_layer()(hidden)

                hidden = self.activation_final(hidden)

        return hidden


class SNMLP(nn.Module):
    """MLP module with Spectral Normalization."""
    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activation_final: Optional[ActivationFn] = None
    bias: bool = True

    @nn.compact
    def __call__(self, data: jax.Array):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = SNDense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                use_bias=self.bias)(hidden)

            if i != len(self.layer_sizes) - 1:
                hidden = self.activation(hidden)
            elif self.activation_final is not None:
                hidden = self.activation_final(hidden)
        return hidden


class VModule(nn.Module):
    """Q Module."""
    hidden_layer_sizes: Sequence[int] = (256, 256)
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(self, obs: jax.Array):
        vs = MLP(layer_sizes=list(self.hidden_layer_sizes) + [1],
                 activation=self.activation,
                 kernel_init=self.kernel_init)(obs)

        return vs.squeeze(-1)


class QModule(nn.Module):
    """Q Module."""
    hidden_layer_sizes: Sequence[int] = (256, 256)
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(self, obs: jax.Array, actions: jax.Array):
        hidden = jnp.concatenate([obs, actions], axis=-1)
        qs = MLP(layer_sizes=list(self.hidden_layer_sizes) + [1],
                 activation=self.activation,
                 kernel_init=self.kernel_init)(hidden)

        return qs.squeeze(-1)


def make_policy_network(
    action_size: int,
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    activation_final: Optional[ActivationFn] = None
) -> nn.Module:
    """Creates a policy network."""
    policy_model = MLP(
        layer_sizes=list(hidden_layer_sizes) + [action_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        activation_final=activation_final)

    def init_fn(rng): return policy_model.init(rng, jnp.zeros((1, obs_size)))

    return policy_model, init_fn


def make_value_network(
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu
) -> nn.Module:
    """Creates a V network: (obs) -> value"""
    value_model = VModule(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )
    dummy_obs = jnp.zeros((1, obs_size))

    def init_fn(rng): return value_model.init(rng, dummy_obs)

    return value_model, init_fn


def make_q_network(
    obs_size: int,
    action_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
) -> nn.Module:
    """Creates a Q network: (obs, action) -> value """

    q_module = QModule(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )
    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))

    def init_fn(rng): return q_module.init(rng, dummy_obs, dummy_action)

    return q_module, init_fn
