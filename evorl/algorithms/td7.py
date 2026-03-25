import logging
from typing import Any, Sequence
import math

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from omegaconf import DictConfig

from evorl.distributed import psum, pmean
from evorl.distributed.gradients import gradient_update
from evorl.envs import AutoresetMode, Box, create_env, Space
from evorl.evaluators import Evaluator
from evorl.metrics import MetricBase, metric_field
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action,
    LossDict,
    Params,
    PolicyExtraInfo,
    PyTreeData,
    PyTreeDict,
    State,
    pytree_field,
)
from evorl.utils import running_statistics
from evorl.utils.jax_utils import (
    scan_and_mean,
    tree_stop_gradient,
    tree_get,
    right_shift_with_padding,
)
from evorl.evaluators import EpisodeCollector
from evorl.agent import Agent, AgentState
from evorl.replay_buffers import LAPReplayBuffer
from evorl.recorders import add_prefix
from evorl.networks.linear import MLP

from .offpolicy_utils import OffPolicyWorkflowTemplate, skip_replay_buffer_state

logger = logging.getLogger(__name__)


def avg_l1_norm(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Average L1 Norm used in TD7."""
    mean_abs = jnp.clip(jnp.mean(jnp.abs(x), axis=-1, keepdims=True), a_min=eps)
    return x / mean_abs


class TD7Encoder(nn.Module):
    z_s_dim: int = 256
    z_sa_dim: int = 256
    f_layer_sizes: Sequence[int] = (256, 256)
    g_layer_sizes: Sequence[int] = (256, 256)

    def setup(self):
        self.zs_mlp = MLP(
            layer_sizes=tuple(self.f_layer_sizes) + (self.z_s_dim,),
            activation=nn.elu,
            name="zs_mlp",
        )
        self.zsa_mlp = MLP(
            layer_sizes=tuple(self.g_layer_sizes) + (self.z_sa_dim,),
            activation=nn.elu,
            name="zsa_mlp",
        )

    def zs(self, obs: jax.Array) -> jax.Array:
        z = self.zs_mlp(obs)
        return avg_l1_norm(z)

    def zsa(self, z_s: jax.Array, action: jax.Array) -> jax.Array:
        z = jnp.concatenate([z_s, action], axis=-1)
        return self.zsa_mlp(z)

    def __call__(
        self, obs: jax.Array, action: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # Utility method to make Flax initialization easier
        z_s = self.zs(obs)
        z_sa = self.zsa(z_s, action)
        return z_s, z_sa


class TD7Actor(nn.Module):
    action_size: int
    z_s_dim: int = 256
    state_emb_dim: int = 256
    hidden_layer_sizes: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs: jax.Array, z_s: jax.Array) -> jax.Array:
        a = nn.Dense(self.state_emb_dim, name="l0")(obs)
        a = avg_l1_norm(a)
        a = jnp.concatenate([a, z_s], axis=-1)

        a = MLP(
            layer_sizes=tuple(self.hidden_layer_sizes) + (self.action_size,),
            activation=nn.relu,
            name="actor_mlp",
        )(a)

        return nn.tanh(a)


class TD7Critic(nn.Module):
    z_s_dim: int = 256
    z_sa_dim: int = 256
    state_action_emb_dim: int = 256
    hidden_layer_sizes: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(
        self, obs: jax.Array, action: jax.Array, z_sa: jax.Array, z_s: jax.Array
    ) -> jax.Array:
        sa = jnp.concatenate([obs, action], axis=-1)

        # q1 network
        q1 = nn.Dense(self.state_action_emb_dim, name="q1_0")(sa)
        q1 = avg_l1_norm(q1)
        q1 = jnp.concatenate([q1, z_sa, z_s], axis=-1)
        q1 = MLP(
            layer_sizes=tuple(self.hidden_layer_sizes) + (1,),
            activation=nn.elu,
            name="q1_mlp",
        )(q1)

        # q2 network
        q2 = nn.Dense(self.state_action_emb_dim, name="q2_0")(sa)
        q2 = avg_l1_norm(q2)
        q2 = jnp.concatenate([q2, z_sa, z_s], axis=-1)
        q2 = MLP(
            layer_sizes=tuple(self.hidden_layer_sizes) + (1,),
            activation=nn.elu,
            name="q2_mlp",
        )(q2)

        return jnp.concatenate([q1, q2], axis=-1)


class TD7TrainMetric(MetricBase):
    critic_loss: chex.Array
    actor_loss: chex.Array
    encoder_loss: chex.Array
    raw_loss_dict: LossDict = metric_field(default_factory=PyTreeDict, reduce_fn=pmean)


class TD7NetworkParams(PyTreeData):
    actor_params: Params
    critic_params: Params
    encoder_params: Params
    target_actor_params: Params
    target_critic_params: Params
    fixed_encoder_params: Params
    fixed_encoder_target_params: Params
    checkpoint_actor_params: Params
    checkpoint_encoder_params: Params


class TD7Agent(Agent):
    """The Agent for TD7."""

    critic_network: nn.Module
    actor_network: nn.Module
    encoder_network: nn.Module
    obs_preprocessor: Any = pytree_field(default=None, static=True)

    discount: float = 0.99
    exploration_epsilon: float = 0.1
    policy_noise: float = 0.2
    clip_policy_noise: float = 0.5
    min_priority: float = 1.0

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        key, q_key, actor_key, enc_key = jax.random.split(key, num=4)

        dummy_obs = jtu.tree_map(lambda x: x[None, ...], obs_space.sample(key))
        dummy_action = action_space.sample(key)[None, ...]

        encoder_params = self.encoder_network.init(enc_key, dummy_obs, dummy_action)

        # Need z_s and z_sa to pass to other networks
        dummy_z_s, dummy_z_sa = self.encoder_network.apply(
            encoder_params, dummy_obs, dummy_action
        )

        critic_params = self.critic_network.init(
            q_key, dummy_obs, dummy_action, dummy_z_sa, dummy_z_s
        )

        actor_params = self.actor_network.init(actor_key, dummy_obs, dummy_z_s)

        params_state = TD7NetworkParams(
            encoder_params=encoder_params,
            actor_params=actor_params,
            critic_params=critic_params,
            fixed_encoder_params=encoder_params,
            target_actor_params=actor_params,
            target_critic_params=critic_params,
            fixed_encoder_target_params=encoder_params,
            checkpoint_actor_params=actor_params,
            checkpoint_encoder_params=encoder_params,
        )

        if self.normalize_obs:
            obs_preprocessor_state = running_statistics.init_state(
                tree_get(dummy_obs, 0)
            )
        else:
            obs_preprocessor_state = None

        # Value clipping states and best performances
        extra_state = PyTreeDict(
            max_q=jnp.array(jnp.finfo(jnp.float32).min, dtype=jnp.float32),
            min_q=jnp.array(jnp.finfo(jnp.float32).max, dtype=jnp.float32),
            max_target=jnp.array(0.0, dtype=jnp.float32),
            min_target=jnp.array(0.0, dtype=jnp.float32),
            best_perf=jnp.array(jnp.finfo(jnp.float32).min, dtype=jnp.float32),
        )

        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state,
            extra_state=extra_state,
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # Uses the fixed_encoder to get the state embedding for the actor
        z_s = self.encoder_network.apply(
            agent_state.params.fixed_encoder_params, obs, method="zs"
        )
        actions = self.actor_network.apply(agent_state.params.actor_params, obs, z_s)

        noise = jax.random.normal(key, actions.shape) * self.exploration_epsilon
        actions += noise
        actions = jnp.clip(actions, -1.0, 1.0)

        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # Evaluation uses checkpoint_encoder and checkpoint_actor
        z_s = self.encoder_network.apply(
            agent_state.params.checkpoint_encoder_params, obs, method="zs"
        )
        actions = self.actor_network.apply(
            agent_state.params.checkpoint_actor_params, obs, z_s
        )

        return actions, PyTreeDict()

    def encoder_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        next_obs = sample_batch.extras.env_extras.ori_obs
        obs = sample_batch.obs
        actions = sample_batch.actions

        if self.normalize_obs:
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        next_z_s = self.encoder_network.apply(
            agent_state.params.encoder_params, next_obs, method="zs"
        )
        next_z_s = jax.lax.stop_gradient(next_z_s)

        z_s = self.encoder_network.apply(
            agent_state.params.encoder_params, obs, method="zs"
        )
        pred_z_sa = self.encoder_network.apply(
            agent_state.params.encoder_params, z_s, actions, method="zsa"
        )

        enc_loss = optax.squared_error(pred_z_sa, next_z_s).mean()
        return PyTreeDict(encoder_loss=enc_loss)

    def critic_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        next_obs = sample_batch.extras.env_extras.ori_obs
        obs = sample_batch.obs
        actions = sample_batch.actions

        if self.normalize_obs:
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # Target Q computation
        fixed_target_z_s = self.encoder_network.apply(
            agent_state.params.fixed_encoder_target_params, next_obs, method="zs"
        )

        noise = jnp.clip(
            jax.random.normal(key, actions.shape) * self.policy_noise,
            -self.clip_policy_noise,
            self.clip_policy_noise,
        )
        next_actions = self.actor_network.apply(
            agent_state.params.target_actor_params, next_obs, fixed_target_z_s
        )
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

        fixed_target_z_sa = self.encoder_network.apply(
            agent_state.params.fixed_encoder_target_params,
            fixed_target_z_s,
            next_actions,
            method="zsa",
        )

        next_qs = self.critic_network.apply(
            agent_state.params.target_critic_params,
            next_obs,
            next_actions,
            fixed_target_z_sa,
            fixed_target_z_s,
        )
        next_qs_min = next_qs.min(-1)

        discounts = self.discount * (1 - sample_batch.extras.env_extras.termination)

        # Value clipping from extra_state
        min_target = agent_state.extra_state.min_target
        max_target = agent_state.extra_state.max_target

        q_target = sample_batch.rewards + discounts * jnp.clip(
            next_qs_min, min_target, max_target
        )
        q_target = jnp.broadcast_to(q_target[..., None], (*q_target.shape, 2))
        q_target = jax.lax.stop_gradient(q_target)

        # Current Q computation
        fixed_z_s = self.encoder_network.apply(
            agent_state.params.fixed_encoder_params, obs, method="zs"
        )
        fixed_z_sa = self.encoder_network.apply(
            agent_state.params.fixed_encoder_params, fixed_z_s, actions, method="zsa"
        )

        qs = self.critic_network.apply(
            agent_state.params.critic_params, obs, actions, fixed_z_sa, fixed_z_s
        )

        td_error = jnp.abs(qs - q_target)

        # LAP huber loss
        critic_loss = (
            jnp.where(
                td_error < self.min_priority,
                0.5 * jnp.square(td_error),
                self.min_priority * td_error,
            )
            .sum(-1)
            .mean()
        )

        # Update running max/min Q values (using global min/max)
        batch_q_max = q_target[..., 0].max()
        batch_q_min = q_target[..., 0].min()

        # Compute priority updates
        priority = jnp.maximum(td_error.max(axis=-1), self.min_priority)

        return PyTreeDict(
            critic_loss=critic_loss,
            q_value=qs.mean(),
            priority=jax.lax.stop_gradient(priority),
            batch_q_max=jax.lax.stop_gradient(batch_q_max),
            batch_q_min=jax.lax.stop_gradient(batch_q_min),
        )

    def actor_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        obs = sample_batch.obs

        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        fixed_z_s = self.encoder_network.apply(
            agent_state.params.fixed_encoder_params, obs, method="zs"
        )
        actions = self.actor_network.apply(
            agent_state.params.actor_params, obs, fixed_z_s
        )

        fixed_z_sa = self.encoder_network.apply(
            agent_state.params.fixed_encoder_params, fixed_z_s, actions, method="zsa"
        )

        qs = self.critic_network.apply(
            agent_state.params.critic_params, obs, actions, fixed_z_sa, fixed_z_s
        )

        actor_loss = -jnp.mean(qs)
        return PyTreeDict(actor_loss=actor_loss)


def make_td7_agent(
    action_space: Space,
    z_s_dim: int = 256,
    z_sa_dim: int = 256,
    f_layer_sizes: Sequence[int] = (256, 256),
    g_layer_sizes: Sequence[int] = (256, 256),
    state_emb_dim: int = 256,
    state_action_emb_dim: int = 256,
    critic_hidden_layer_sizes: Sequence[int] = (256, 256),
    actor_hidden_layer_sizes: Sequence[int] = (256, 256),
    discount: float = 0.99,
    exploration_epsilon: float = 0.1,
    policy_noise: float = 0.2,
    clip_policy_noise: float = 0.5,
    min_priority: float = 1.0,
    normalize_obs: bool = False,
):
    assert isinstance(action_space, Box), "Only continue action space is supported."

    action_size = action_space.shape[0]

    encoder_network = TD7Encoder(
        z_s_dim=z_s_dim,
        z_sa_dim=z_sa_dim,
        f_layer_sizes=f_layer_sizes,
        g_layer_sizes=g_layer_sizes,
    )
    critic_network = TD7Critic(
        z_s_dim=z_s_dim,
        z_sa_dim=z_sa_dim,
        state_action_emb_dim=state_action_emb_dim,
        hidden_layer_sizes=critic_hidden_layer_sizes,
    )
    actor_network = TD7Actor(
        z_s_dim=z_s_dim,
        state_emb_dim=state_emb_dim,
        hidden_layer_sizes=actor_hidden_layer_sizes,
        action_size=action_size,
    )

    if normalize_obs:
        obs_preprocessor = running_statistics.normalize
    else:
        obs_preprocessor = None

    return TD7Agent(
        encoder_network=encoder_network,
        critic_network=critic_network,
        actor_network=actor_network,
        obs_preprocessor=obs_preprocessor,
        discount=discount,
        exploration_epsilon=exploration_epsilon,
        policy_noise=policy_noise,
        clip_policy_noise=clip_policy_noise,
        min_priority=min_priority,
    )


class TD7Workflow(OffPolicyWorkflowTemplate):
    @classmethod
    def name(cls):
        return "TD7"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        assert config.rollout_episodes % config.num_envs == 0, (
            "rollout_episodes must be divisible by num_envs"
        )

        env = create_env(
            config.env,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
            record_ori_obs=True,
        )

        agent = make_td7_agent(
            action_space=env.action_space,
            z_s_dim=config.agent_network.zs_dim,
            z_sa_dim=config.agent_network.zsa_dim,
            f_layer_sizes=config.agent_network.f_layer_sizes,
            g_layer_sizes=config.agent_network.g_layer_sizes,
            state_emb_dim=config.agent_network.state_emb_dim,
            state_action_emb_dim=config.agent_network.state_action_emb_dim,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            min_priority=config.min_priority,
            normalize_obs=config.normalize_obs,
        )

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

        replay_buffer = LAPReplayBuffer(
            capacity=config.replay_buffer_capacity,
            sample_batch_size=config.batch_size,
            alpha=config.lap_alpha,
        )

        eval_env = create_env(
            config.env,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        collector = EpisodeCollector(
            env=env,
            action_fn=agent.compute_actions,
            max_episode_steps=config.env.max_episode_steps,
            env_extra_fields=("ori_obs", "termination"),
        )

        workflow = cls(
            env,
            agent,
            optimizer,
            evaluator,
            replay_buffer,
            config,
        )
        workflow.collector = collector
        return workflow

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree]:
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)

        opt_state = PyTreeDict(
            actor=self.optimizer.init(agent_state.params.actor_params),
            critic=self.optimizer.init(agent_state.params.critic_params),
            encoder=self.optimizer.init(agent_state.params.encoder_params),
        )
        return agent_state, opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        # Evaluate via episodes (rollout [T, episodes, ...])
        eval_metrics, trajectory = self.collector.rollout(
            state.agent_state, rollout_key, self.config.rollout_episodes
        )

        trajectory = trajectory.replace(next_obs=None)

        # Mask out padded steps based on `dones` array (since autoreset is OFF)
        mask = jnp.logical_not(right_shift_with_padding(trajectory.dones, 1))
        trajectory = trajectory.replace(dones=None)

        def _flatten_fn(x):
            return x.reshape(-1, *x.shape[2:])

        trajectory = jtu.tree_map(_flatten_fn, trajectory)
        mask = jtu.tree_map(_flatten_fn, mask)

        trajectory, mask = tree_stop_gradient((trajectory, mask))

        agent_state = state.agent_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    dp_axis_name=self.dp_axis_name,
                )
            )

        replay_buffer_state = self.replay_buffer.add(
            state.replay_buffer_state, trajectory, mask=mask
        )

        # Gradient update wrappers (can't easily use agent_gradient_update because params are nested in TD7NetworkParams)
        def encoder_loss_fn(params, agent_state, sample_batch, key):
            # Evaluate using modified encoder params
            temp_agent_state = agent_state.replace(
                params=agent_state.params.replace(encoder_params=params)
            )
            loss_dict = self.agent.encoder_loss(temp_agent_state, sample_batch, key)
            return loss_dict.encoder_loss, loss_dict

        def critic_loss_fn(params, agent_state, sample_batch, key):
            temp_agent_state = agent_state.replace(
                params=agent_state.params.replace(critic_params=params)
            )
            loss_dict = self.agent.critic_loss(temp_agent_state, sample_batch, key)
            return loss_dict.critic_loss, loss_dict

        def actor_loss_fn(params, agent_state, sample_batch, key):
            temp_agent_state = agent_state.replace(
                params=agent_state.params.replace(actor_params=params)
            )
            loss_dict = self.agent.actor_loss(temp_agent_state, sample_batch, key)
            return loss_dict.actor_loss, loss_dict

        encoder_update_fn = gradient_update(
            encoder_loss_fn,
            self.optimizer,
            dp_axis_name=self.dp_axis_name,
            has_aux=True,
        )

        critic_update_fn = gradient_update(
            critic_loss_fn,
            self.optimizer,
            dp_axis_name=self.dp_axis_name,
            has_aux=True,
        )

        actor_update_fn = gradient_update(
            actor_loss_fn,
            self.optimizer,
            dp_axis_name=self.dp_axis_name,
            has_aux=True,
        )

        def _sample_and_update_fn(carry, t):
            key, agent_state, opt_state, replay_state, training_steps = carry

            enc_opt_state = opt_state.encoder
            critic_opt_state = opt_state.critic
            actor_opt_state = opt_state.actor

            key, enc_key, critic_key, actor_key, rb_key = jax.random.split(key, num=5)

            # Sample from Replay Buffer (yields batch + index tracking state in one pass)
            sample_batch, _weights, replay_state = self.replay_buffer.sample(
                replay_state, rb_key
            )

            # 1. Update Encoder
            (enc_loss, enc_loss_dict), enc_params, enc_opt_state = encoder_update_fn(
                enc_opt_state,
                agent_state.params.encoder_params,
                agent_state,
                sample_batch,
                enc_key,
            )
            agent_state = agent_state.replace(
                params=agent_state.params.replace(encoder_params=enc_params)
            )

            # 2. Update Critic
            (critic_loss, critic_loss_dict), critic_params, critic_opt_state = (
                critic_update_fn(
                    critic_opt_state,
                    agent_state.params.critic_params,
                    agent_state,
                    sample_batch,
                    critic_key,
                )
            )
            agent_state = agent_state.replace(
                params=agent_state.params.replace(critic_params=critic_params)
            )

            # LAP Priority updates
            priority = critic_loss_dict.priority
            replay_state = self.replay_buffer.update_priority(replay_state, priority)

            # Value clipping min/max tracking updates
            new_max = jnp.maximum(
                agent_state.extra_state.max_q, critic_loss_dict.batch_q_max
            )
            new_min = jnp.minimum(
                agent_state.extra_state.min_q, critic_loss_dict.batch_q_min
            )
            agent_state = agent_state.replace(
                extra_state=agent_state.extra_state.replace(
                    max_q=new_max,
                    min_q=new_min,
                )
            )

            # 3. Update Actor
            def _update_actor(carry):
                agent_state, actor_opt_state = carry
                (actor_loss, actor_loss_dict), actor_params, actor_opt_state = (
                    actor_update_fn(
                        actor_opt_state,
                        agent_state.params.actor_params,
                        agent_state,
                        sample_batch,
                        actor_key,
                    )
                )
                agent_state = agent_state.replace(
                    params=agent_state.params.replace(actor_params=actor_params)
                )
                return agent_state, actor_opt_state, actor_loss, actor_loss_dict

            def _skip_actor(carry):
                agent_state, actor_opt_state = carry
                return (
                    agent_state,
                    actor_opt_state,
                    jnp.array(0.0),
                    PyTreeDict(actor_loss=jnp.array(0.0)),
                )

            agent_state, actor_opt_state, actor_loss, actor_loss_dict = jax.lax.cond(
                (training_steps + 1) % self.config.policy_freq == 0,
                _update_actor,
                _skip_actor,
                (agent_state, actor_opt_state),
            )

            # 4. Hard target updates
            def _hard_target_updates(carry):
                agent_state, replay_state = carry
                agent_state = agent_state.replace(
                    params=agent_state.params.replace(
                        target_actor_params=agent_state.params.actor_params,
                        target_critic_params=agent_state.params.critic_params,
                        fixed_encoder_target_params=agent_state.params.fixed_encoder_params,
                        fixed_encoder_params=agent_state.params.encoder_params,
                    ),
                    extra_state=agent_state.extra_state.replace(
                        max_target=agent_state.extra_state.max_q,
                        min_target=agent_state.extra_state.min_q,
                    ),
                )
                replay_state = self.replay_buffer.reset_max_priority(replay_state)
                return agent_state, replay_state

            def _skip_updates(carry):
                return carry

            agent_state, replay_state = jax.lax.cond(
                (training_steps + 1) % self.config.target_update_rate == 0,
                _hard_target_updates,
                _skip_updates,
                (agent_state, replay_state),
            )

            opt_state = opt_state.replace(
                encoder=enc_opt_state, actor=actor_opt_state, critic=critic_opt_state
            )

            # We use zero for dummy actor losses if we didn't update it to avoid NaN downstream
            return (
                (key, agent_state, opt_state, replay_state, training_steps + 1),
                (
                    enc_loss,
                    critic_loss,
                    actor_loss,
                    enc_loss_dict,
                    critic_loss_dict,
                    actor_loss_dict,
                ),
            )

        # Retrieve global training steps from state metrics iterations
        global_steps = state.metrics.iterations * self.config.num_updates_per_iter

        # Need to cast loop dummy variable to integer
        iters = jnp.arange(self.config.num_updates_per_iter, dtype=jnp.int32)

        (
            (_, agent_state, opt_state, replay_buffer_state, _),
            (
                encoder_loss,
                critic_loss,
                actor_loss,
                enc_loss_dict,
                critic_loss_dict,
                actor_loss_dict,
            ),
        ) = scan_and_mean(
            _sample_and_update_fn,
            (
                learn_key,
                agent_state,
                state.opt_state,
                replay_buffer_state,
                global_steps,
            ),
            iters,
            length=self.config.num_updates_per_iter,
        )

        # Episodic Checkpointing evaluate & replace
        if self.config.checkpoint_metric == "mean":
            perf = jnp.mean(eval_metrics.episode_returns)
        elif self.config.checkpoint_metric == "min":
            perf = jnp.min(eval_metrics.episode_returns)
        elif self.config.checkpoint_metric == "max":
            perf = jnp.max(eval_metrics.episode_returns)
        else:
            raise ValueError(
                f"Unsupported checkpoint metric: {self.config.checkpoint_metric}. "
                "Must be one of 'min', 'max', or 'mean'."
            )

        def _update_checkpoint(ag_state):
            return ag_state.replace(
                params=ag_state.params.replace(
                    checkpoint_actor_params=ag_state.params.actor_params,
                    checkpoint_encoder_params=ag_state.params.fixed_encoder_params,
                ),
                extra_state=ag_state.extra_state.replace(best_perf=perf),
            )

        agent_state = jax.lax.cond(
            perf >= agent_state.extra_state.best_perf,
            _update_checkpoint,
            lambda ag_state: ag_state,
            agent_state,
        )

        # actor loss would be divided by policy_freq effectively (thanks to zeros)
        # So multiply back by policy_freq to get the real mean
        actor_loss = actor_loss * self.config.policy_freq

        train_metrics = TD7TrainMetric(
            encoder_loss=encoder_loss,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            raw_loss_dict=PyTreeDict(
                {**enc_loss_dict, **critic_loss_dict, **actor_loss_dict}
            ),
        ).all_reduce(dp_axis_name=self.dp_axis_name)

        sampled_timesteps = jnp.uint32(eval_metrics.episode_lengths.sum())
        sampled_timesteps = psum(sampled_timesteps, axis_name=self.dp_axis_name)

        sampled_epsiodes = psum(
            jnp.uint32(self.config.rollout_episodes), axis_name=self.dp_axis_name
        )

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_epsiodes,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(dp_axis_name=self.dp_axis_name)

        # state.env_state is irrelevant here since episodecollector handles reset internally.
        # we can just pass original env_state unmodified since disabled autoreset env ignores it generally across steps (or we just maintain step 0).
        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
            opt_state=opt_state,
        )

    def learn(self, state: State) -> State:
        num_devices = jax.device_count()
        one_step_episodes = self.config.rollout_episodes * num_devices
        sampled_episodes = state.metrics.sampled_episodes.tolist()
        num_iters = math.ceil(
            (self.config.total_episodes - sampled_episodes)
            / (one_step_episodes * self.config.fold_iters)
        )
        start_iteration = state.metrics.iterations.tolist()
        final_iteration = num_iters + start_iteration

        for i in range(start_iteration, final_iteration):
            iterations = i + 1
            train_metrics, state = self._multi_steps(state)
            workflow_metrics = state.metrics

            self.recorder.write(train_metrics.to_local_dict(), iterations)
            self.recorder.write(workflow_metrics.to_local_dict(), iterations)

            if (
                iterations % self.config.eval_interval == 0
                or iterations == final_iteration
            ):
                eval_metrics, state = self.evaluate(state)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iterations
                )

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)
            self.checkpoint_manager.save(
                iterations, saved_state, force=iterations == final_iteration
            )

        return state
