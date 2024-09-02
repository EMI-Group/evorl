from collections.abc import Sequence
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn

from evorl.agent import AgentState
from evorl.envs import Space
from evorl.networks import MLP, ActivationFn, Initializer
from evorl.types import (
    pytree_field,
)
from evorl.utils import running_statistics

from ..td3 import TD3Agent, TD3NetworkParams


def make_policy_network(
    pop_size: int,
    action_size: int,
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    activation_final: ActivationFn | None = None,
) -> nn.Module:
    """Creates a batched policy network."""
    policy_model = MLP(
        layer_sizes=tuple(hidden_layer_sizes) + (action_size,),
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        activation_final=activation_final,
        norm_layer=nn.LayerNorm,
    )

    def init_fn(rng):
        return policy_model.init(rng, jnp.zeros((1, obs_size)))

    return policy_model, init_fn


def make_q_network(
    obs_size: int,
    action_size: int,
    n_stack: int = 1,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
) -> nn.Module:
    """Creates a Q network with LayerNorm: (obs, action) -> value"""

    class QModule(nn.Module):
        """Q Module."""

        n: int

        @nn.compact
        def __call__(self, obs: jax.Array, actions: jax.Array):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            if self.n == 1:
                qs = MLP(
                    layer_sizes=tuple(hidden_layer_sizes) + (1,),
                    activation=activation,
                    kernel_init=kernel_init,
                    norm_layer=nn.LayerNorm,
                )(hidden)
            elif self.n > 1:
                hidden = jnp.broadcast_to(hidden, (self.n,) + hidden.shape)
                qs = nn.vmap(
                    MLP,
                    out_axes=-2,
                    variable_axes={"params": 0},
                    split_rngs={"params": True},
                )(
                    layer_sizes=tuple(hidden_layer_sizes) + (1,),
                    activation=activation,
                    kernel_init=kernel_init,
                    norm_layer=nn.LayerNorm,
                )(hidden)
            else:
                raise ValueError("n should be greater than 0")

            return qs.squeeze(-1)

    q_module = QModule(n=n_stack)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))

    def init_fn(rng):
        return q_module.init(rng, dummy_obs, dummy_action)

    return q_module, init_fn


def _create_agent_state_pytree_axes():
    return AgentState(
        params=TD3NetworkParams(
            critic_params=None,
            actor_params=0,
            target_critic_params=None,
            target_actor_params=0,
        ),
        obs_preprocessor_state=None,
        action_space=None,
    )


class PopTD3Agent(TD3Agent):
    pop_size: int = 1
    agent_state_pytree_axes: chex.ArrayTree = pytree_field(
        default_factory=_create_agent_state_pytree_axes, pytree_node=False
    )

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        obs_size = obs_space.shape[0]
        action_size = action_space.shape[0]

        key, critic_key, actor_key, obs_preprocessor_key = jax.random.split(key, 4)

        # global critic network
        # the output of the q_network is (b, n_critics), n_critics is the number of critics, b is the batch size
        critic_network, critic_init_fn = make_q_network(
            obs_size=obs_size,
            action_size=action_size,
            n_stack=2,
            hidden_layer_sizes=self.critic_hidden_layer_sizes,
        )
        critic_params = critic_init_fn(critic_key)
        target_critic_params = critic_params

        # pop actor networks
        # the output of the actor_network is (b,), b is the batch size
        actor_network, actor_init_fn = make_policy_network(
            pop_size=self.pop_size,
            action_size=action_size,
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes,
            activation_final=nn.tanh,
        )

        pop_actor_params = jax.vmap(actor_init_fn)(
            jax.random.split(actor_key, self.pop_size)
        )
        target_pop_actor_params = pop_actor_params

        self.set_frozen_attr("critic_network", critic_network)
        self.set_frozen_attr("actor_network", actor_network)

        params_state = TD3NetworkParams(
            critic_params=critic_params,
            actor_params=pop_actor_params,
            target_critic_params=target_critic_params,
            target_actor_params=target_pop_actor_params,
        )

        # shared obs_preprocessor
        if self.normalize_obs:
            obs_preprocessor = running_statistics.normalize
            self.set_frozen_attr("obs_preprocessor", obs_preprocessor)
            dummy_obs = obs_space.sample(obs_preprocessor_key)
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state,
            action_space=action_space,
        )
