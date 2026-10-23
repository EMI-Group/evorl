"""Native NNX multilayer perceptrons and policy/value networks."""

from collections.abc import Callable, Sequence
from typing import Any
from itertools import pairwise

import jax
import jax.numpy as jnp
from flax import nnx

from .layer_norm import get_norm_layer
from .spectral_norm import SNDense

ActivationFn = Callable[[jax.Array], jax.Array]
Initializer = Callable[..., Any]


class MLP(nnx.Module):
    """Fully connected network with normalization on hidden layers only.

    Args:
        input_size: Input feature dimension.
        layer_sizes: Output dimension of each linear layer, including the output.
        activation: Activation applied after hidden layers.
        kernel_init: Initializer used for each linear kernel.
        activation_final: Optional output activation.
        use_bias: Whether the linear layers have biases.
        norm_layer: Optional normalization class for hidden layers.
        rngs: Random streams used to initialize variables.
    """

    def __init__(
        self,
        input_size: int,
        layer_sizes: Sequence[int],
        activation: ActivationFn = jax.nn.relu,
        kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
        activation_final: ActivationFn | None = None,
        use_bias: bool = True,
        norm_layer: type[nnx.Module] | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.activation = activation
        self.activation_final = activation_final
        sizes = (input_size, *layer_sizes)
        self.layers = nnx.List(
            nnx.Linear(
                in_size,
                out_size,
                kernel_init=kernel_init,
                use_bias=use_bias,
                rngs=rngs,
            )
            for in_size, out_size in pairwise(sizes)
        )
        self.layer_norms = nnx.List(
            norm_layer(size, rngs=rngs) if norm_layer else None
            for size in layer_sizes[:-1]
        )

    def __call__(self, data: jax.Array) -> jax.Array:
        hidden = data
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if i < len(self.layers) - 1:
                if self.layer_norms[i] is not None:
                    hidden = self.layer_norms[i](hidden)
                hidden = self.activation(hidden)
            elif self.activation_final is not None:
                hidden = self.activation_final(hidden)
        return hidden


class SNMLP(nnx.Module):
    """MLP with spectral normalization on every linear layer.

    Singular vectors are non-parameter state. Each call advances power iteration
    on the local model, and callers retain it by extracting the updated state.
    """

    def __init__(
        self,
        input_size: int,
        layer_sizes: Sequence[int],
        activation: ActivationFn = jax.nn.relu,
        kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
        activation_final: ActivationFn | None = None,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.activation = activation
        self.activation_final = activation_final
        sizes = (input_size, *layer_sizes)
        self.layers = nnx.List(
            SNDense(
                in_size,
                out_size,
                kernel_init=kernel_init,
                use_bias=use_bias,
                rngs=rngs,
            )
            for in_size, out_size in pairwise(sizes)
        )

    def __call__(self, data: jax.Array) -> jax.Array:
        hidden = data
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if i < len(self.layers) - 1:
                hidden = self.activation(hidden)
            elif self.activation_final is not None:
                hidden = self.activation_final(hidden)
        return hidden


def make_mlp(
    input_size: int,
    layer_sizes: Sequence[int],
    activation: ActivationFn = jax.nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    activation_final: ActivationFn | None = None,
    use_bias: bool = True,
    norm_layer_type: str = "none",
    *,
    rngs: nnx.Rngs,
) -> MLP | SNMLP:
    """Construct an MLP with explicit input dimensions and initialized state."""
    kwargs = dict(
        input_size=input_size,
        layer_sizes=layer_sizes,
        activation=activation,
        kernel_init=kernel_init,
        activation_final=activation_final,
        use_bias=use_bias,
        rngs=rngs,
    )
    if norm_layer_type == "spectral_norm":
        return SNMLP(**kwargs)
    return MLP(**kwargs, norm_layer=get_norm_layer(norm_layer_type))


class EnsembleMLP(nnx.Module):
    """Independent MLPs with variables stacked on their leading dimension.

    Inputs have shape `(num_networks, ..., input_size)`; the network axis in the
    output is placed at `out_axes`. Parameters and spectral state are mapped
    together, so no network shares weights or power-iteration state.
    """

    def __init__(
        self,
        input_size: int,
        layer_sizes: Sequence[int],
        num_networks: int,
        *,
        out_axes: int = -2,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        if num_networks < 1:
            raise ValueError("num_networks must be positive")
        self.num_networks = num_networks
        self.out_axes = out_axes

        @nnx.split_rngs(splits=num_networks)
        @nnx.vmap(in_axes=0, out_axes=0)
        def create(rngs):
            return make_mlp(input_size, layer_sizes, rngs=rngs, **kwargs)

        self.mlp = create(rngs)

    def __call__(self, data: jax.Array) -> jax.Array:
        if data.shape[0] != self.num_networks:
            raise ValueError("Input leading dimension must equal num_networks")
        return nnx.vmap(
            lambda model, x: model(x), in_axes=(0, 0), out_axes=self.out_axes
        )(self.mlp, data)


def make_vmap_mlp(
    input_size: int,
    layer_sizes: Sequence[int],
    num_networks: int,
    activation: ActivationFn = jax.nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    activation_final: ActivationFn | None = None,
    use_bias: bool = True,
    norm_layer_type: str = "none",
    out_axes: int = -2,
    *,
    rngs: nnx.Rngs,
) -> EnsembleMLP:
    """Construct independent MLPs with stacked parameters and mapped inputs."""
    return EnsembleMLP(
        input_size,
        layer_sizes,
        num_networks,
        activation=activation,
        kernel_init=kernel_init,
        activation_final=activation_final,
        use_bias=use_bias,
        norm_layer_type=norm_layer_type,
        out_axes=out_axes,
        rngs=rngs,
    )


class PolicyNetwork(nnx.Module):
    """Policy MLP, optionally selecting a field from dictionary observations."""

    def __init__(self, mlp: nnx.Module, obs_key: str = ""):
        self.mlp = mlp
        self.obs_key = obs_key

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self.mlp(obs[self.obs_key] if self.obs_key else obs)


class ValueNetwork(PolicyNetwork):
    """Scalar state-value network without a trailing singleton dimension."""

    def __call__(self, obs: jax.Array) -> jax.Array:
        return super().__call__(obs).squeeze(-1)


class QNetwork(nnx.Module):
    """Continuous-action critic with an optional axis of independent critics."""

    def __init__(self, mlp: nnx.Module, n_stack: int, obs_key: str = ""):
        self.mlp = mlp
        self.n_stack = n_stack
        self.obs_key = obs_key

    def __call__(self, obs: jax.Array, actions: jax.Array) -> jax.Array:
        obs = obs[self.obs_key] if self.obs_key else obs
        inputs = jnp.concatenate((obs, actions), axis=-1)
        if self.n_stack > 1:
            inputs = jnp.broadcast_to(inputs, (self.n_stack, *inputs.shape))
        return self.mlp(inputs).squeeze(-1)


class DiscreteQNetwork(nnx.Module):
    """Action values, optionally with a penultimate independent-critic axis."""

    def __init__(self, mlp: nnx.Module, n_stack: int, obs_key: str = ""):
        self.mlp = mlp
        self.n_stack = n_stack
        self.obs_key = obs_key

    def __call__(self, obs: jax.Array) -> jax.Array:
        inputs = obs[self.obs_key] if self.obs_key else obs
        if self.n_stack > 1:
            inputs = jnp.broadcast_to(inputs, (self.n_stack, *inputs.shape))
        return self.mlp(inputs)


class MultiObjectiveQNetwork(QNetwork):
    """Continuous critic retaining its final objective axis, even for one objective.

    Outputs have shape `(..., objectives)` or `(..., critics, objectives)`.
    """

    def __call__(self, obs: jax.Array, actions: jax.Array) -> jax.Array:
        obs = obs[self.obs_key] if self.obs_key else obs
        inputs = jnp.concatenate((obs, actions), axis=-1)
        if self.n_stack > 1:
            inputs = jnp.broadcast_to(inputs, (self.n_stack, *inputs.shape))
        return self.mlp(inputs)


class MultiObjectiveDiscreteQNetwork(DiscreteQNetwork):
    """Discrete critic with separate action and objective axes.

    Outputs have shape `(..., actions, objectives)` or
    `(..., critics, actions, objectives)`.
    """

    def __init__(
        self,
        mlp: nnx.Module,
        n_stack: int,
        num_objectives: int,
        obs_key: str = "",
    ):
        super().__init__(mlp, n_stack, obs_key)
        self.num_objectives = num_objectives

    def __call__(self, obs: jax.Array) -> jax.Array:
        values = super().__call__(obs)
        return values.reshape(
            (
                *values.shape[:-1],
                values.shape[-1] // self.num_objectives,
                self.num_objectives,
            )
        )


def make_policy_network(
    input_size: int,
    action_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    use_bias: bool = True,
    activation: ActivationFn = jax.nn.relu,
    activation_final: ActivationFn | None = None,
    norm_layer_type: str = "none",
    obs_key: str = "",
    *,
    rngs: nnx.Rngs,
) -> PolicyNetwork:
    """Create a policy mapping observation features to action parameters."""
    return PolicyNetwork(
        make_mlp(
            input_size,
            (*hidden_layer_sizes, action_size),
            activation=activation,
            activation_final=activation_final,
            use_bias=use_bias,
            norm_layer_type=norm_layer_type,
            rngs=rngs,
        ),
        obs_key,
    )


def make_v_network(
    input_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = jax.nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    norm_layer_type: str = "none",
    obs_key: str = "",
    *,
    rngs: nnx.Rngs,
) -> ValueNetwork:
    """Create a scalar value function preserving all observation batch axes."""
    return ValueNetwork(
        make_mlp(
            input_size,
            (*hidden_layer_sizes, 1),
            activation=activation,
            kernel_init=kernel_init,
            norm_layer_type=norm_layer_type,
            rngs=rngs,
        ),
        obs_key,
    )


def make_q_network(
    input_size: int,
    action_size: int,
    n_stack: int = 1,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = jax.nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    norm_layer_type: str = "none",
    obs_key: str = "",
    *,
    rngs: nnx.Rngs,
) -> QNetwork:
    """Create a continuous critic; `input_size` counts observation features.

    One critic returns `(...)`; multiple critics return `(..., n_stack)`.
    """
    if n_stack < 1:
        raise ValueError("n_stack must be positive")
    kwargs = dict(
        input_size=input_size + action_size,
        layer_sizes=(*hidden_layer_sizes, 1),
        activation=activation,
        kernel_init=kernel_init,
        norm_layer_type=norm_layer_type,
        rngs=rngs,
    )
    mlp = (
        make_vmap_mlp(**kwargs, num_networks=n_stack)
        if n_stack > 1
        else make_mlp(**kwargs)
    )
    return QNetwork(mlp, n_stack, obs_key)


def make_discrete_q_network(
    input_size: int,
    action_size: int,
    n_stack: int = 1,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = jax.nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    norm_layer_type: str = "none",
    obs_key: str = "",
    *,
    rngs: nnx.Rngs,
) -> DiscreteQNetwork:
    """Create critics with output `(..., actions)` or `(..., critics, actions)`."""
    if n_stack < 1:
        raise ValueError("n_stack must be positive")
    kwargs = dict(
        input_size=input_size,
        layer_sizes=(*hidden_layer_sizes, action_size),
        activation=activation,
        kernel_init=kernel_init,
        norm_layer_type=norm_layer_type,
        rngs=rngs,
    )
    mlp = (
        make_vmap_mlp(**kwargs, num_networks=n_stack)
        if n_stack > 1
        else make_mlp(**kwargs)
    )
    return DiscreteQNetwork(mlp, n_stack, obs_key)


def make_multi_objective_q_network(
    input_size: int,
    action_size: int,
    num_objectives: int,
    n_stack: int = 1,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = jax.nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    norm_layer_type: str = "none",
    obs_key: str = "",
    *,
    rngs: nnx.Rngs,
) -> MultiObjectiveQNetwork:
    """Create vector critics; `input_size` counts observation features.

    Outputs retain the objective axis: `(..., num_objectives)` for one critic,
    or `(..., n_stack, num_objectives)` for multiple independent critics.
    """
    if n_stack < 1:
        raise ValueError("n_stack must be positive")
    if num_objectives < 1:
        raise ValueError("num_objectives must be positive")
    kwargs = dict(
        input_size=input_size + action_size,
        layer_sizes=(*hidden_layer_sizes, num_objectives),
        activation=activation,
        kernel_init=kernel_init,
        norm_layer_type=norm_layer_type,
        rngs=rngs,
    )
    mlp = (
        make_vmap_mlp(**kwargs, num_networks=n_stack)
        if n_stack > 1
        else make_mlp(**kwargs)
    )
    return MultiObjectiveQNetwork(mlp, n_stack, obs_key)


def make_multi_objective_discrete_q_network(
    input_size: int,
    action_size: int,
    num_objectives: int,
    n_stack: int = 1,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = jax.nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    norm_layer_type: str = "none",
    obs_key: str = "",
    *,
    rngs: nnx.Rngs,
) -> MultiObjectiveDiscreteQNetwork:
    """Create vector action values with a final `num_objectives` axis.

    Outputs have shape `(..., action_size, num_objectives)` for one critic,
    or `(..., n_stack, action_size, num_objectives)` for independent critics.
    """
    if n_stack < 1:
        raise ValueError("n_stack must be positive")
    if num_objectives < 1:
        raise ValueError("num_objectives must be positive")
    kwargs = dict(
        input_size=input_size,
        layer_sizes=(*hidden_layer_sizes, action_size * num_objectives),
        activation=activation,
        kernel_init=kernel_init,
        norm_layer_type=norm_layer_type,
        rngs=rngs,
    )
    mlp = (
        make_vmap_mlp(**kwargs, num_networks=n_stack)
        if n_stack > 1
        else make_mlp(**kwargs)
    )
    return MultiObjectiveDiscreteQNetwork(mlp, n_stack, num_objectives, obs_key)
