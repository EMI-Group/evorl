from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from .spectral_norm import SNDense
from .layer_norm import get_norm_layer

ActivationFn = Callable[[jax.Array], jax.Array]
Initializer = Callable[..., Any]


class MLP(nn.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activation_final: ActivationFn | None = None
    use_bias: bool = True
    norm_layer: nn.Module | None = None

    @nn.compact
    def __call__(self, data: jax.Array):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.use_bias,
            )(hidden)

            if i != len(self.layer_sizes) - 1:
                if self.norm_layer is not None:
                    hidden = self.norm_layer()(hidden)

                hidden = self.activation(hidden)
            elif self.activation_final is not None:
                # if self.norm_layer is not None:
                #     hidden = self.norm_layer()(hidden)

                hidden = self.activation_final(hidden)

        return hidden


class SNMLP(nn.Module):
    """MLP module with Spectral Normalization."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activation_final: ActivationFn | None = None
    use_bias: bool = True

    @nn.compact
    def __call__(self, data: jax.Array):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = SNDense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.use_bias,
            )(hidden)

            if i != len(self.layer_sizes) - 1:
                hidden = self.activation(hidden)
            elif self.activation_final is not None:
                hidden = self.activation_final(hidden)
        return hidden


def make_mlp(
    layer_sizes: Sequence[int],
    activation: ActivationFn = nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    activation_final: ActivationFn | None = None,
    use_bias: bool = True,
    norm_layer_type: str = "none",
) -> nn.Module:
    """Creates an MLP network."""
    if norm_layer_type == "spectral_norm":
        mlp = SNMLP(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=kernel_init,
            activation_final=activation_final,
            use_bias=use_bias,
        )
    else:
        mlp = MLP(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=kernel_init,
            activation_final=activation_final,
            use_bias=use_bias,
            norm_layer=get_norm_layer(norm_layer_type),
        )

    return mlp


def make_vmap_mlp(
    layer_sizes: Sequence[int],
    activation: ActivationFn = nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    activation_final: ActivationFn | None = None,
    use_bias: bool = True,
    norm_layer_type: str = "none",
    out_axes: int = -2,
):
    """Creates multiple MLP networks in parallel."""
    if norm_layer_type == "spectral_norm":
        mlp = nn.vmap(
            SNMLP,
            out_axes=out_axes,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=kernel_init,
            activation_final=activation_final,
            use_bias=use_bias,
        )
    else:
        mlp = nn.vmap(
            MLP,
            out_axes=out_axes,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=kernel_init,
            activation_final=activation_final,
            norm_layer=get_norm_layer(norm_layer_type),
            use_bias=use_bias,
        )

    return mlp


def make_policy_network(
    action_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    use_bias: bool = True,
    activation: ActivationFn = nn.relu,
    activation_final: ActivationFn | None = None,
    norm_layer_type: str = "none",
    obs_key: str = "",
) -> nn.Module:
    """Creates a policy network."""

    class PolicyModule(nn.Module):
        @nn.compact
        def __call__(self, obs: jax.Array):
            if obs_key:
                obs = obs[obs_key]

            actions = make_mlp(
                layer_sizes=tuple(hidden_layer_sizes) + (action_size,),
                activation=activation,
                kernel_init=jax.nn.initializers.lecun_uniform(),
                activation_final=activation_final,
                use_bias=use_bias,
                norm_layer_type=norm_layer_type,
            )(obs)

            return actions

    policy_model = PolicyModule()

    return policy_model


def make_v_network(
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    norm_layer_type: str = "none",
    obs_key: str = "",
) -> nn.Module:
    """Creates a V network: (obs) -> value."""

    class VModule(nn.Module):
        @nn.compact
        def __call__(self, obs: jax.Array):
            if obs_key:
                obs = obs[obs_key]

            vs = make_mlp(
                layer_sizes=tuple(hidden_layer_sizes) + (1,),
                activation=activation,
                kernel_init=kernel_init,
                norm_layer_type=norm_layer_type,
            )(obs)

            return vs.squeeze(-1)

    value_model = VModule()

    return value_model


def make_q_network(
    n_stack: int = 1,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    norm_layer_type: str = "none",
    obs_key: str = "",
) -> nn.Module:
    """Creates a Q network: (obs, action) -> value."""

    class QModule(nn.Module):
        """Q Module for continuous action space."""

        n: int

        @nn.compact
        def __call__(self, obs: jax.Array, actions: jax.Array):
            if obs_key:
                obs = obs[obs_key]

            hidden = jnp.concatenate([obs, actions], axis=-1)
            if self.n == 1:
                qs = make_mlp(
                    layer_sizes=tuple(hidden_layer_sizes) + (1,),
                    activation=activation,
                    kernel_init=kernel_init,
                    norm_layer_type=norm_layer_type,
                )(hidden)
            elif self.n > 1:
                hidden = jnp.broadcast_to(hidden, (self.n,) + hidden.shape)
                qs = make_vmap_mlp(
                    layer_sizes=tuple(hidden_layer_sizes) + (1,),
                    activation=activation,
                    kernel_init=kernel_init,
                    norm_layer_type=norm_layer_type,
                )(hidden)
            else:
                raise ValueError("n should be greater than 0")

            return qs.squeeze(-1)

    q_module = QModule(n=n_stack)

    return q_module


def make_discrete_q_network(
    action_size: int,
    n_stack: int = 1,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    norm_layer_type: str = "none",
    obs_key: str = "",
) -> nn.Module:
    """Creates a Q network for discrete action space: (obs) -> q_values."""

    class QModule(nn.Module):
        """Q Module for discrete action space."""

        n: int

        @nn.compact
        def __call__(self, obs: jax.Array):
            if obs_key:
                obs = obs[obs_key]

            if self.n == 1:
                qs = make_mlp(
                    layer_sizes=tuple(hidden_layer_sizes) + (action_size,),
                    activation=activation,
                    kernel_init=kernel_init,
                    norm_layer_type=norm_layer_type,
                )(obs)
            elif self.n > 1:
                obs = jnp.broadcast_to(obs, (self.n,) + obs.shape)
                qs = make_vmap_mlp(
                    layer_sizes=tuple(hidden_layer_sizes) + (action_size,),
                    activation=activation,
                    kernel_init=kernel_init,
                    norm_layer_type=norm_layer_type,
                )(obs)
            else:
                raise ValueError("n should be greater than 0")

            return qs

    q_module = QModule(n=n_stack)

    return q_module
