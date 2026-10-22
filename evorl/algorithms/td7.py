from evorl.utils import running_statistics
import logging
from typing import Any
from collections.abc import Sequence
import math

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from omegaconf import DictConfig

from evorl.distributed import pmax, pmean, pmin, psum
from evorl.distributed.gradients import pmean_nnx_loss_and_grad
from evorl.distributed.sharding import mark_varying
from evorl.envs import Box, AutoresetMode, create_env, Space
from evorl.metrics import MetricBase, metric_field
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action,
    LossDict,
    PolicyExtraInfo,
    PyTreeDict,
    State,
    pytree_field,
)
from evorl.utils.jax_utils import (
    scan_and_mean,
    tree_stop_gradient,
    right_shift_with_padding,
)
from evorl.evaluators import Evaluator, EpisodeCollector
from evorl.agent import Agent, AgentState
from evorl.replay_buffers import LAPReplayBuffer
from evorl.recorders import add_prefix
from evorl.networks.linear import MLP

from .offpolicy_utils import OffPolicyWorkflowTemplate, skip_replay_buffer_state

logger = logging.getLogger(__name__)


def avg_l1_norm(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Average L1 Norm used in TD7."""
    mean_abs = jnp.clip(jnp.mean(jnp.abs(x), axis=-1, keepdims=True), min=eps)
    return x / mean_abs


class TD7Encoder(nnx.Module):
    """TD7 state and state-action embeddings with independent MLPs."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        z_s_dim: int = 256,
        z_sa_dim: int = 256,
        f_layer_sizes: Sequence[int] = (256, 256),
        g_layer_sizes: Sequence[int] = (256, 256),
        *,
        rngs: nnx.Rngs,
    ):
        self.zs_mlp = MLP(
            obs_size,
            (*f_layer_sizes, z_s_dim),
            activation=jax.nn.elu,
            rngs=rngs,
        )
        self.zsa_mlp = MLP(
            z_s_dim + action_size,
            (*g_layer_sizes, z_sa_dim),
            activation=jax.nn.elu,
            rngs=rngs,
        )

    def zs(self, obs: jax.Array) -> jax.Array:
        return avg_l1_norm(self.zs_mlp(obs))

    def zsa(self, z_s: jax.Array, action: jax.Array) -> jax.Array:
        return self.zsa_mlp(jnp.concatenate((z_s, action), axis=-1))

    def __call__(
        self, obs: jax.Array, action: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        z_s = self.zs(obs)
        return z_s, self.zsa(z_s, action)


class TD7Actor(nnx.Module):
    """TD7 actor combining a normalized observation projection and state embedding."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        z_s_dim: int = 256,
        state_emb_dim: int = 256,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        *,
        rngs: nnx.Rngs,
    ):
        # Linen Dense used LeCun normal here, unlike the MLP's uniform kernels.
        self.l0 = nnx.Linear(obs_size, state_emb_dim, rngs=rngs)
        self.actor_mlp = MLP(
            state_emb_dim + z_s_dim,
            (*hidden_layer_sizes, action_size),
            activation=jax.nn.relu,
            rngs=rngs,
        )

    def __call__(self, obs: jax.Array, z_s: jax.Array) -> jax.Array:
        features = jnp.concatenate((avg_l1_norm(self.l0(obs)), z_s), axis=-1)
        return jnp.tanh(self.actor_mlp(features))


class TD7Critic(nnx.Module):
    """Independent twin critics using state, action and encoder embeddings."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        z_s_dim: int = 256,
        z_sa_dim: int = 256,
        state_action_emb_dim: int = 256,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        *,
        rngs: nnx.Rngs,
    ):
        self.q1_0 = nnx.Linear(obs_size + action_size, state_action_emb_dim, rngs=rngs)
        self.q2_0 = nnx.Linear(obs_size + action_size, state_action_emb_dim, rngs=rngs)
        input_size = state_action_emb_dim + z_sa_dim + z_s_dim
        self.q1_mlp = MLP(
            input_size, (*hidden_layer_sizes, 1), activation=jax.nn.elu, rngs=rngs
        )
        self.q2_mlp = MLP(
            input_size, (*hidden_layer_sizes, 1), activation=jax.nn.elu, rngs=rngs
        )

    def __call__(
        self, obs: jax.Array, action: jax.Array, z_sa: jax.Array, z_s: jax.Array
    ) -> jax.Array:
        state_action = jnp.concatenate((obs, action), axis=-1)
        q1 = self.q1_mlp(
            jnp.concatenate((avg_l1_norm(self.q1_0(state_action)), z_sa, z_s), axis=-1)
        )
        q2 = self.q2_mlp(
            jnp.concatenate((avg_l1_norm(self.q2_0(state_action)), z_sa, z_s), axis=-1)
        )
        return jnp.concatenate((q1, q2), axis=-1)


class TD7TrainMetric(MetricBase):
    # Primary losses are already averaged by pmean_loss_and_grad.
    critic_loss: jax.Array
    actor_loss: jax.Array
    encoder_loss: jax.Array
    raw_loss_dict: LossDict = metric_field(default_factory=PyTreeDict, reduce_fn=pmean)


class TD7Agent(Agent):
    """The Agent for TD7."""

    agent_network_cfg: DictConfig = pytree_field(static=True)

    obs_preprocessor: Any = pytree_field(default=None, static=True)

    discount: float = 0.99
    exploration_epsilon: float = 0.1
    policy_noise: float = 0.2
    clip_policy_noise: float = 0.5
    min_priority: float = 1.0

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        if not isinstance(action_space, Box):
            raise TypeError(f"Unsupported action space: {action_space}")
        key, q_key, actor_key, enc_key = jax.random.split(key, num=4)

        cfg = self.agent_network_cfg
        encoder = TD7Encoder(
            obs_size=obs_space.shape[-1],
            action_size=action_space.shape[0],
            z_s_dim=cfg.zs_dim,
            z_sa_dim=cfg.zsa_dim,
            f_layer_sizes=tuple(cfg.f_layer_sizes),
            g_layer_sizes=tuple(cfg.g_layer_sizes),
            rngs=nnx.Rngs(enc_key),
        )
        actor = TD7Actor(
            obs_size=obs_space.shape[-1],
            action_size=action_space.shape[0],
            z_s_dim=cfg.zs_dim,
            state_emb_dim=cfg.state_emb_dim,
            hidden_layer_sizes=tuple(cfg.actor_hidden_layer_sizes),
            rngs=nnx.Rngs(actor_key),
        )
        critic = TD7Critic(
            obs_size=obs_space.shape[-1],
            action_size=action_space.shape[0],
            z_s_dim=cfg.zs_dim,
            z_sa_dim=cfg.zsa_dim,
            state_action_emb_dim=cfg.state_action_emb_dim,
            hidden_layer_sizes=tuple(cfg.critic_hidden_layer_sizes),
            rngs=nnx.Rngs(q_key),
        )
        models = nnx.Dict(
            actor=actor,
            critic=critic,
            encoder=encoder,
            target_actor=nnx.clone(actor),
            target_critic=nnx.clone(critic),
            fixed_encoder=nnx.clone(encoder),
            fixed_encoder_target=nnx.clone(encoder),
            checkpoint_actor=nnx.clone(actor),
            checkpoint_encoder=nnx.clone(encoder),
        )

        if self.normalize_obs:
            obs_preprocessor_state = running_statistics.init_state(
                obs_space.sample(key)
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
            model_state=nnx.unpack(models),
            obs_preprocessor_state=obs_preprocessor_state,
            extra_state=extra_state,
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        models = self.get_models(agent_state, copy=True)
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # Uses the fixed_encoder to get the state embedding for the actor
        z_s = models.fixed_encoder.zs(obs)
        actions = models.actor(obs, z_s)

        noise = jax.random.normal(key, actions.shape) * self.exploration_epsilon
        actions += noise
        actions = jnp.clip(actions, -1.0, 1.0)

        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        models = self.get_models(agent_state, copy=True)
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # Evaluation uses checkpoint_encoder and checkpoint_actor
        z_s = models.checkpoint_encoder.zs(obs)
        actions = models.checkpoint_actor(obs, z_s)

        return actions, PyTreeDict()

    def encoder_loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        models = self.get_models(agent_state)
        next_obs = sample_batch.extras.env_extras.ori_obs
        obs = sample_batch.obs
        actions = sample_batch.actions

        if self.normalize_obs:
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        next_z_s = models.encoder.zs(next_obs)
        next_z_s = jax.lax.stop_gradient(next_z_s)

        z_s = models.encoder.zs(obs)
        pred_z_sa = models.encoder.zsa(z_s, actions)

        enc_loss = optax.squared_error(pred_z_sa, next_z_s).mean()
        return PyTreeDict(encoder_loss=enc_loss)

    def critic_loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        models = self.get_models(agent_state)
        next_obs = sample_batch.extras.env_extras.ori_obs
        obs = sample_batch.obs
        actions = sample_batch.actions

        if self.normalize_obs:
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # Target Q computation
        fixed_target_z_s = models.fixed_encoder_target.zs(next_obs)

        noise = jnp.clip(
            jax.random.normal(key, actions.shape) * self.policy_noise,
            -self.clip_policy_noise,
            self.clip_policy_noise,
        )
        next_actions = models.target_actor(next_obs, fixed_target_z_s)
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

        fixed_target_z_sa = models.fixed_encoder_target.zsa(
            fixed_target_z_s, next_actions
        )

        next_qs = models.target_critic(
            next_obs, next_actions, fixed_target_z_sa, fixed_target_z_s
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
        fixed_z_s = models.fixed_encoder.zs(obs)
        fixed_z_sa = models.fixed_encoder.zsa(fixed_z_s, actions)

        qs = models.critic(obs, actions, fixed_z_sa, fixed_z_s)

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
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        models = self.get_models(agent_state)
        obs = sample_batch.obs

        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        fixed_z_s = models.fixed_encoder.zs(obs)
        actions = models.actor(obs, fixed_z_s)

        fixed_z_sa = models.fixed_encoder.zsa(fixed_z_s, actions)

        qs = models.critic(obs, actions, fixed_z_sa, fixed_z_s)

        actor_loss = -jnp.mean(qs)
        return PyTreeDict(actor_loss=actor_loss)


class TD7Workflow(OffPolicyWorkflowTemplate):
    @classmethod
    def name(cls):
        return "TD7"

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        """Split the configured total rollout episodes across devices."""
        super()._rescale_config(config)
        num_devices = jax.local_device_count()
        if config.rollout_episodes < num_devices:
            raise ValueError(
                f"rollout_episodes({config.rollout_episodes}) must be at least num_devices({num_devices})"
            )
        if config.rollout_episodes % num_devices != 0:
            logger.warning(
                f"rollout_episodes({config.rollout_episodes}) cannot be divided by num_devices({num_devices}), "
                f"rescale rollout_episodes to {config.rollout_episodes // num_devices}"
            )
        config.rollout_episodes = config.rollout_episodes // num_devices

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        assert config.rollout_episodes % config.num_envs == 0, (
            "rollout_episodes must be divisible by num_envs"
        )

        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
            record_ori_obs=True,
        )

        agent = TD7Agent(
            agent_network_cfg=config.agent_network,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            min_priority=config.min_priority,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
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
        self, key: jax.Array
    ) -> tuple[AgentState, chex.ArrayTree]:
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        models = nnx.merge(agent_state.model_state, copy=True)
        opt_state = PyTreeDict(
            actor=nnx.unpack(
                nnx.Optimizer(models.actor, self.optimizer, wrt=nnx.Param)
            ),
            critic=nnx.unpack(
                nnx.Optimizer(models.critic, self.optimizer, wrt=nnx.Param)
            ),
            encoder=nnx.unpack(
                nnx.Optimizer(models.encoder, self.optimizer, wrt=nnx.Param)
            ),
        )
        return agent_state, opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)
        if self.is_multi_device:
            rank = jax.lax.axis_index(self.dp_axis_name)
            rollout_key = jax.random.fold_in(rollout_key, rank)
            learn_key = jax.random.fold_in(learn_key, rank)

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
                    axis_name=self.dp_axis_name,
                )
            )

        replay_buffer_state = self.replay_buffer.add(
            state.replay_buffer_state, trajectory, mask=mask
        )

        def _update_encoder(agent_state, opt_state, sample_batch, key):
            """Apply one encoder update and return new model and optimizer snapshots."""
            models = nnx.merge(agent_state.model_state, copy=True)

            def loss_fn(models, batch, key):
                losses = self.agent.encoder_loss(
                    agent_state.replace(model_state=nnx.unpack(models)), batch, key
                )
                return losses.encoder_loss, losses

            value, grads = pmean_nnx_loss_and_grad(
                loss_fn,
                self.dp_axis_name,
                has_aux=True,
                wrt=nnx.All(nnx.Param, nnx.PathContains("encoder")),
            )(models, sample_batch, key)
            optimizer = nnx.merge(opt_state, copy=True)
            optimizer.update(models.encoder, grads["encoder"])
            return (
                value,
                agent_state.replace(model_state=nnx.unpack(models)),
                nnx.unpack(optimizer),
            )

        def _update_critic(agent_state, opt_state, sample_batch, key):
            """Apply one critic update and return new model and optimizer snapshots."""
            models = nnx.merge(agent_state.model_state, copy=True)

            def loss_fn(models, batch, key):
                losses = self.agent.critic_loss(
                    agent_state.replace(model_state=nnx.unpack(models)), batch, key
                )
                return losses.critic_loss, losses

            value, grads = pmean_nnx_loss_and_grad(
                loss_fn,
                self.dp_axis_name,
                has_aux=True,
                wrt=nnx.All(nnx.Param, nnx.PathContains("critic")),
            )(models, sample_batch, key)
            optimizer = nnx.merge(opt_state, copy=True)
            optimizer.update(models.critic, grads["critic"])
            return (
                value,
                agent_state.replace(model_state=nnx.unpack(models)),
                nnx.unpack(optimizer),
            )

        def _update_actor(agent_state, opt_state, sample_batch, key):
            """Apply one actor update and return new model and optimizer snapshots."""
            models = nnx.merge(agent_state.model_state, copy=True)

            def loss_fn(models, batch, key):
                losses = self.agent.actor_loss(
                    agent_state.replace(model_state=nnx.unpack(models)), batch, key
                )
                return losses.actor_loss, losses

            value, grads = pmean_nnx_loss_and_grad(
                loss_fn,
                self.dp_axis_name,
                has_aux=True,
                wrt=nnx.All(nnx.Param, nnx.PathContains("actor")),
            )(models, sample_batch, key)
            optimizer = nnx.merge(opt_state, copy=True)
            optimizer.update(models.actor, grads["actor"])
            return (
                value,
                agent_state.replace(model_state=nnx.unpack(models)),
                nnx.unpack(optimizer),
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
            (enc_loss, enc_loss_dict), agent_state, enc_opt_state = _update_encoder(
                agent_state,
                enc_opt_state,
                sample_batch,
                enc_key,
            )

            # 2. Update Critic
            (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
                _update_critic(
                    agent_state,
                    critic_opt_state,
                    sample_batch,
                    critic_key,
                )
            )

            # LAP Priority updates
            priority = critic_loss_dict.priority
            replay_state = self.replay_buffer.update_priority(replay_state, priority)

            # Value clipping min/max tracking updates
            new_max = pmax(
                jnp.maximum(
                    agent_state.extra_state.max_q, critic_loss_dict.batch_q_max
                ),
                axis_name=self.dp_axis_name,
            )
            new_min = pmin(
                jnp.minimum(
                    agent_state.extra_state.min_q, critic_loss_dict.batch_q_min
                ),
                axis_name=self.dp_axis_name,
            )
            agent_state = agent_state.replace(
                extra_state=agent_state.extra_state.replace(
                    max_q=new_max,
                    min_q=new_min,
                )
            )

            # 3. Update Actor
            def _update_actor_step(carry):
                agent_state, actor_opt_state = carry
                (actor_loss, actor_loss_dict), agent_state, actor_opt_state = (
                    _update_actor(
                        agent_state,
                        actor_opt_state,
                        sample_batch,
                        actor_key,
                    )
                )
                return agent_state, actor_opt_state, actor_loss, actor_loss_dict

            def _skip_actor(carry):
                agent_state, actor_opt_state = carry
                actor_loss = jnp.array(0.0)
                local_actor_loss = mark_varying(
                    actor_loss, jax.sharding.get_abstract_mesh().manual_axes
                )
                if not self.is_multi_device:
                    actor_loss = local_actor_loss
                return (
                    agent_state,
                    actor_opt_state,
                    actor_loss,
                    PyTreeDict(actor_loss=local_actor_loss),
                )

            agent_state, actor_opt_state, actor_loss, actor_loss_dict = jax.lax.cond(
                (training_steps + 1) % self.config.policy_freq == 0,
                _update_actor_step,
                _skip_actor,
                (agent_state, actor_opt_state),
            )

            # 4. Hard target updates
            def _hard_target_updates(carry):
                agent_state, replay_state = carry
                models = nnx.merge(agent_state.model_state, copy=True)
                nnx.update(models.target_actor, nnx.state(models.actor, nnx.Param))
                nnx.update(models.target_critic, nnx.state(models.critic, nnx.Param))
                nnx.update(
                    models.fixed_encoder_target,
                    nnx.state(models.fixed_encoder, nnx.Param),
                )
                nnx.update(models.fixed_encoder, nnx.state(models.encoder, nnx.Param))
                agent_state = agent_state.replace(
                    model_state=nnx.unpack(models),
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
            perf = pmean(
                jnp.mean(eval_metrics.episode_returns), axis_name=self.dp_axis_name
            )
        elif self.config.checkpoint_metric == "min":
            perf = pmin(
                jnp.min(eval_metrics.episode_returns), axis_name=self.dp_axis_name
            )
        elif self.config.checkpoint_metric == "max":
            perf = pmax(
                jnp.max(eval_metrics.episode_returns), axis_name=self.dp_axis_name
            )
        else:
            raise ValueError(
                f"Unsupported checkpoint metric: {self.config.checkpoint_metric}. "
                "Must be one of 'min', 'max', or 'mean'."
            )

        def _update_checkpoint(ag_state):
            models = nnx.merge(ag_state.model_state, copy=True)
            nnx.update(models.checkpoint_actor, nnx.state(models.actor, nnx.Param))
            nnx.update(
                models.checkpoint_encoder, nnx.state(models.fixed_encoder, nnx.Param)
            )
            return ag_state.replace(
                model_state=nnx.unpack(models),
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
        ).all_reduce(axis_name=self.dp_axis_name)

        sampled_timesteps = jnp.uint32(eval_metrics.episode_lengths.sum())
        sampled_timesteps = psum(sampled_timesteps, axis_name=self.dp_axis_name)

        sampled_epsiodes = psum(
            jnp.uint32(self.config.rollout_episodes), axis_name=self.dp_axis_name
        )

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_epsiodes,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(axis_name=self.dp_axis_name)

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
        num_devices = self.num_devices
        one_step_episodes = self.config.rollout_episodes * num_devices
        sampled_episodes = state.metrics.sampled_episodes.tolist()
        num_fold_iters = math.ceil(
            (self.config.total_episodes - sampled_episodes)
            / (one_step_episodes * self.config.fold_iters)
        )
        start_iteration = state.metrics.iterations.tolist()
        final_iteration = start_iteration + num_fold_iters * self.config.fold_iters

        for i in range(num_fold_iters):
            train_metrics, state = self._multi_steps(state)
            workflow_metrics = state.metrics
            iterations = workflow_metrics.iterations.tolist()

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
