import logging
from collections.abc import Sequence

import chex
import flashbax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig

from evorl.networks import MLP, ActivationFn, Initializer, StaticLayerNorm
from evorl.envs import AutoresetMode, Box, create_env, Space
from evorl.evaluator import Evaluator
from evorl.utils import running_statistics

from evorl.agent import AgentState

from ..td3 import TD3Agent, TD3NetworkParams, TD3Workflow

logger = logging.getLogger(__name__)


def make_policy_network(
    action_size: int,
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    activation_final: ActivationFn | None = None,
    static_layer_norm: bool = False,
) -> nn.Module:
    norm_layer = StaticLayerNorm if static_layer_norm else nn.LayerNorm

    """Creates a batched policy network."""
    policy_model = MLP(
        layer_sizes=tuple(hidden_layer_sizes) + (action_size,),
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        activation_final=activation_final,
        norm_layer=norm_layer,
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
    static_layer_norm: bool = False,
) -> nn.Module:
    """Creates a Q network with LayerNorm: (obs, action) -> value"""
    norm_layer = StaticLayerNorm if static_layer_norm else nn.LayerNorm

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
                    norm_layer=norm_layer,
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
                    norm_layer=norm_layer,
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


class TD3LayerNormAgent(TD3Agent):
    static_layer_norm: bool = False

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        obs_size = obs_space.shape[0]
        action_size = action_space.shape[0]

        key, critic_key, actor_key, obs_preprocessor_key = jax.random.split(key, num=4)

        # the output of the q_network is (b, n_critics), n_critics is the number of critics, b is the batch size
        critic_network, critic_init_fn = make_q_network(
            obs_size=obs_size,
            action_size=action_size,
            n_stack=2,
            hidden_layer_sizes=self.critic_hidden_layer_sizes,
            static_layer_norm=self.static_layer_norm,
        )
        critic_params = critic_init_fn(critic_key)
        target_critic_params = critic_params

        # the output of the actor_network is (b,), b is the batch size
        actor_network, actor_init_fn = make_policy_network(
            action_size=action_size,
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes,
            activation_final=nn.tanh,
            static_layer_norm=self.static_layer_norm,
        )

        actor_params = actor_init_fn(actor_key)
        target_actor_params = actor_params

        self.set_frozen_attr("critic_network", critic_network)
        self.set_frozen_attr("actor_network", actor_network)

        params_state = TD3NetworkParams(
            critic_params=critic_params,
            actor_params=actor_params,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
        )

        # obs_preprocessor
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


class TD3LayerNormWorkflow(TD3Workflow):
    @classmethod
    def name(cls):
        return "TD3-LayerNorm"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        """
        return workflow
        """
        env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
        )

        assert isinstance(
            env.action_space, Box
        ), "Only continue action space is supported."

        agent = TD3LayerNormAgent(
            static_layer_norm=config.agent_network.static_layer_norm,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
        )

        # one optimizer, two opt_states (in setup function) for both actor and critic
        if (
            config.optimizer.grad_clip_norm is not None
            and config.optimizer.grad_clip_norm > 0
        ):
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.optimizer.grad_clip_norm),
                optax.adam(config.optimizer.lr),
            )
        else:
            optimizer = optax.adam(config.optimizer.lr)

        replay_buffer = flashbax.make_item_buffer(
            max_length=config.replay_buffer_capacity,
            min_length=max(config.batch_size, config.learning_start_timesteps),
            sample_batch_size=config.batch_size,
            add_batches=True,
        )

        eval_env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env, agent=agent, max_episode_steps=config.env.max_episode_steps
        )

        return cls(
            env,
            agent,
            optimizer,
            evaluator,
            replay_buffer,
            config,
        )
