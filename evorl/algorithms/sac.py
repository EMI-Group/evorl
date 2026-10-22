from evorl.utils import running_statistics
import logging
from typing import Any
from omegaconf import DictConfig

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import optax

from evorl.replay_buffers import ReplayBuffer
from evorl.distributed import psum, pmean
from evorl.distributed.gradients import pmean_nnx_loss_and_grad
from evorl.distribution import get_tanh_norm_dist, get_categorical_dist
from evorl.envs import Discrete, Box, AutoresetMode, create_env, Space
from evorl.evaluators import Evaluator
from evorl.metrics import MetricBase, metric_field
from evorl.networks import make_policy_network, make_q_network, make_discrete_q_network
from evorl.rollout import rollout
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action,
    LossDict,
    PolicyExtraInfo,
    PyTreeDict,
    State,
    pytree_field,
)
from evorl.utils.jax_utils import scan_and_mean, tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update

from evorl.agent import Agent, AgentState
from .offpolicy_utils import OffPolicyWorkflowTemplate, clean_trajectory

logger = logging.getLogger(__name__)


class SACTrainMetric(MetricBase):
    # Primary losses are already averaged by pmean_nnx_loss_and_grad.
    critic_loss: jax.Array
    actor_loss: jax.Array
    alpha_loss: jax.Array | None = None
    raw_loss_dict: LossDict = metric_field(default_factory=PyTreeDict, reduce_fn=pmean)


class SACAgent(Agent):
    agent_network_cfg: DictConfig = pytree_field(static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)

    init_alpha: float = 1.0
    discount: float = 0.99

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        if not isinstance(action_space, Box):
            raise TypeError(f"Unsupported action space: {action_space}")
        if self.agent_network_cfg.get("num_critics", 2) < 2:
            raise ValueError("num_critics must be at least 2 for twin-critic training")
        key, critic_key, actor_key = jax.random.split(key, num=3)

        cfg = self.agent_network_cfg
        policy_obs_key = cfg.get("policy_obs_key")
        value_obs_key = cfg.get("value_obs_key")
        actor = make_policy_network(
            input_size=(
                obs_space.spaces[policy_obs_key] if policy_obs_key else obs_space
            ).shape[-1],
            action_size=action_space.shape[0] * 2,
            hidden_layer_sizes=tuple(cfg.actor_hidden_layer_sizes),
            obs_key=policy_obs_key,
            rngs=nnx.Rngs(actor_key),
        )
        critic = make_q_network(
            input_size=(
                obs_space.spaces[value_obs_key] if value_obs_key else obs_space
            ).shape[-1],
            action_size=action_space.shape[0],
            n_stack=cfg.get("num_critics", 2),
            hidden_layer_sizes=tuple(cfg.critic_hidden_layer_sizes),
            obs_key=value_obs_key,
            rngs=nnx.Rngs(critic_key),
        )
        models = nnx.Dict(
            actor=actor,
            critic=critic,
            target_critic=nnx.clone(critic),
            log_alpha=nnx.Param(jnp.log(jnp.float32(self.init_alpha))),
        )

        if self.normalize_obs:
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(
                obs_space.sample(key)
            )
        else:
            obs_preprocessor_state = None

        target_entropy = -jnp.prod(jnp.array(action_space.shape, dtype=jnp.float32))

        return AgentState(
            model_state=nnx.unpack(models),
            obs_preprocessor_state=obs_preprocessor_state,
            extra_state=PyTreeDict(target_entropy=target_entropy),  # the constant
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.get_models(agent_state, copy=True).actor(obs)
        actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        actions = actions_dist.sample(seed=key)
        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.get_models(agent_state, copy=True).actor(obs)
        actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        actions = actions_dist.mode()
        return actions, PyTreeDict()

    def alpha_loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        models = self.get_models(agent_state)
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = models.actor(obs)
        actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        actions = actions_dist.sample(seed=key)
        actions_logp = actions_dist.log_prob(actions)

        # official impl:
        alpha = jnp.exp(models.log_alpha[...])
        alpha_loss = jnp.mean(
            -alpha
            * jax.lax.stop_gradient(
                actions_logp + agent_state.extra_state.target_entropy
            )
        )

        # another impl: see stable-baselines3/issues/36
        # alpha_loss = (- models.log_alpha[...] *
        #               jax.lax.stop_gradient(actions_logp + target_entropy)).mean()

        return PyTreeDict(
            alpha_loss=alpha_loss, log_alpha=models.log_alpha[...], alpha=alpha
        )

    def actor_loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        models = self.get_models(agent_state)
        actor_key, entropy_key = jax.random.split(key, 2)
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        alpha = jnp.exp(models.log_alpha[...])

        raw_actions = models.actor(obs)
        actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        actions = actions_dist.sample(seed=actor_key)
        actions_logp = actions_dist.log_prob(actions)

        # [B, 2]
        qs = models.critic(obs, actions)
        qs_min = jnp.min(qs, axis=-1)
        actor_loss = jnp.mean(alpha * actions_logp - qs_min)
        entropy = actions_dist.entropy(seed=entropy_key).mean()

        return PyTreeDict(actor_loss=actor_loss, entropy=entropy)

    def critic_loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        models = self.get_models(agent_state)
        obs = sample_batch.obs
        next_obs = sample_batch.extras.env_extras.ori_obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )

        discounts = self.discount * (1 - sample_batch.extras.env_extras.termination)

        alpha = jnp.exp(models.log_alpha[...])

        # [B, 2]
        qs = models.critic(obs, sample_batch.actions)

        next_raw_actions = models.actor(next_obs)
        next_actions_dist = get_tanh_norm_dist(*jnp.split(next_raw_actions, 2, axis=-1))
        next_actions = next_actions_dist.sample(seed=key)
        next_actions_logp = next_actions_dist.log_prob(next_actions)
        # [B, 2]
        next_qs = models.target_critic(next_obs, next_actions)
        qs_target = sample_batch.rewards + discounts * (
            jnp.min(next_qs, axis=-1) - alpha * next_actions_logp
        )
        qs_target = jax.lax.stop_gradient(
            jnp.broadcast_to(qs_target[..., None], qs.shape)
        )

        q_loss = optax.squared_error(qs, qs_target).sum(-1).mean()
        return PyTreeDict(critic_loss=q_loss)


class SACDiscreteAgent(Agent):
    agent_network_cfg: DictConfig = pytree_field(static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)

    init_alpha: float = 1.0
    discount: float = 0.99
    target_entropy_ratio: float = 0.98

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        if not isinstance(action_space, Discrete):
            raise TypeError(f"Unsupported action space: {action_space}")
        if self.agent_network_cfg.get("num_critics", 2) < 2:
            raise ValueError("num_critics must be at least 2 for twin-critic training")
        key, critic_key, actor_key = jax.random.split(key, num=3)

        cfg = self.agent_network_cfg
        policy_obs_key = cfg.get("policy_obs_key")
        value_obs_key = cfg.get("value_obs_key")
        actor = make_policy_network(
            input_size=(
                obs_space.spaces[policy_obs_key] if policy_obs_key else obs_space
            ).shape[-1],
            action_size=action_space.n,
            hidden_layer_sizes=tuple(cfg.actor_hidden_layer_sizes),
            obs_key=policy_obs_key,
            rngs=nnx.Rngs(actor_key),
        )
        critic = make_discrete_q_network(
            input_size=(
                obs_space.spaces[value_obs_key] if value_obs_key else obs_space
            ).shape[-1],
            action_size=action_space.n,
            n_stack=cfg.get("num_critics", 2),
            hidden_layer_sizes=tuple(cfg.critic_hidden_layer_sizes),
            obs_key=value_obs_key,
            rngs=nnx.Rngs(critic_key),
        )
        models = nnx.Dict(
            actor=actor,
            critic=critic,
            target_critic=nnx.clone(critic),
            log_alpha=nnx.Param(jnp.log(jnp.float32(self.init_alpha))),
        )

        if self.normalize_obs:
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(
                obs_space.sample(key)
            )
        else:
            obs_preprocessor_state = None

        target_entropy = self.target_entropy_ratio * jnp.log(
            jnp.float32(action_space.n)
        )

        return AgentState(
            model_state=nnx.unpack(models),
            obs_preprocessor_state=obs_preprocessor_state,
            extra_state=PyTreeDict(target_entropy=target_entropy),  # the constant
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.get_models(agent_state, copy=True).actor(obs)
        actions_dist = get_categorical_dist(raw_actions)
        actions = actions_dist.sample(seed=key)
        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.get_models(agent_state, copy=True).actor(obs)
        actions_dist = get_categorical_dist(raw_actions)
        actions = actions_dist.mode()
        return actions, PyTreeDict()

    def alpha_loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        models = self.get_models(agent_state)
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = models.actor(obs)
        actions_dist = get_categorical_dist(raw_actions)
        entropy = actions_dist.entropy()

        # official impl:
        alpha = jnp.exp(models.log_alpha[...])
        alpha_loss = -jnp.mean(
            alpha
            * jax.lax.stop_gradient(agent_state.extra_state.target_entropy - entropy)
        )

        return PyTreeDict(
            alpha_loss=alpha_loss,
        )

    def actor_loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        models = self.get_models(agent_state)
        actor_key, entropy_key = jax.random.split(key, 2)
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        alpha = jnp.exp(models.log_alpha[...])

        raw_actions = models.actor(obs)
        actions_dist = get_categorical_dist(raw_actions)
        entropy = actions_dist.entropy()
        actions_prob = jax.nn.softmax(raw_actions)

        # [B, 2, n]
        qs = models.critic(obs)
        qs_min = jnp.min(qs, axis=-2)
        qs_estimate = jnp.sum(qs_min * actions_prob, axis=-1)
        actor_loss = -jnp.mean(alpha * entropy + qs_estimate)

        return PyTreeDict(actor_loss=actor_loss, entropy=entropy.mean())

    def critic_loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        models = self.get_models(agent_state)
        obs = sample_batch.obs
        next_obs = sample_batch.extras.env_extras.ori_obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )

        discounts = self.discount * (1 - sample_batch.extras.env_extras.termination)

        alpha = jnp.exp(models.log_alpha[...])

        # [B, 2, n]
        qs = models.critic(obs)
        qs = jnp.take_along_axis(
            qs,
            sample_batch.actions.reshape(-1, 1, 1),
            axis=-1,
        ).squeeze(-1)

        next_raw_actions = models.actor(next_obs)
        next_actions_prob = jax.nn.softmax(next_raw_actions)
        next_actions_logp = jax.nn.log_softmax(next_raw_actions)
        # [B, 2, n]
        next_qs = models.target_critic(next_obs)
        next_qs_min = jnp.min(next_qs, axis=-2)  # [B, n]
        next_qs_estimate = jnp.sum(
            next_actions_prob * (next_qs_min - alpha * next_actions_logp), axis=-1
        )  # [B]

        qs_target = sample_batch.rewards + discounts * next_qs_estimate
        qs_target = jax.lax.stop_gradient(
            jnp.broadcast_to(qs_target[..., None], qs.shape)
        )

        q_loss = optax.squared_error(qs, qs_target).sum(-1).mean()
        return PyTreeDict(critic_loss=q_loss, q_value=qs.mean())


class SACWorkflow(OffPolicyWorkflowTemplate):
    @classmethod
    def name(cls):
        return "SAC"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )

        if isinstance(env.action_space, Box):
            agent = SACAgent(
                agent_network_cfg=config.agent_network,
                init_alpha=config.alpha,
                discount=config.discount,
                obs_preprocessor=running_statistics.normalize
                if config.normalize_obs
                else None,
            )
        elif isinstance(env.action_space, Discrete):
            agent = SACDiscreteAgent(
                agent_network_cfg=config.agent_network,
                init_alpha=config.alpha,
                discount=config.discount,
                obs_preprocessor=running_statistics.normalize
                if config.normalize_obs
                else None,
                target_entropy_ratio=config.target_entropy_ratio,
            )
        else:
            raise TypeError(f"Unsupported action space: {env.action_space}")

        # TODO: use different lr for critic and actor
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

        replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            min_sample_timesteps=max(
                config.batch_size, config.learning_start_timesteps
            ),
            sample_batch_size=config.batch_size,
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

        return cls(
            env,
            agent,
            optimizer,
            evaluator,
            replay_buffer,
            config,
        )

    def _setup_agent_and_optimizer(
        self, key: jax.Array
    ) -> tuple[AgentState, chex.ArrayTree]:
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        models = nnx.merge(agent_state.model_state, copy=True)
        opt_state = PyTreeDict(
            dict(
                actor=nnx.unpack(
                    nnx.Optimizer(models.actor, self.optimizer, wrt=nnx.Param)
                ),
                critic=nnx.unpack(
                    nnx.Optimizer(models.critic, self.optimizer, wrt=nnx.Param)
                ),
            )
        )
        if self.config.adaptive_alpha:
            opt_state = opt_state.replace(
                alpha=nnx.unpack(
                    nnx.Optimizer(
                        nnx.Dict(log_alpha=models.log_alpha),
                        self.optimizer,
                        wrt=nnx.Param,
                    )
                )
            )

        return agent_state, opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)
        if self.is_multi_device:
            rank = jax.lax.axis_index(self.dp_axis_name)
            rollout_key = jax.random.fold_in(rollout_key, rank)
            learn_key = jax.random.fold_in(learn_key, rank)

        # the trajectory [T, B, ...]
        trajectory, env_state = rollout(
            env_fn=self.env.step,
            action_fn=self.agent.compute_actions,
            env_state=state.env_state,
            agent_state=state.agent_state,
            key=rollout_key,
            rollout_length=self.config.rollout_length,
            env_extra_fields=("ori_obs", "termination"),
        )

        trajectory_dones = trajectory.dones
        trajectory = clean_trajectory(trajectory)
        trajectory = flatten_rollout_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)

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
            state.replay_buffer_state, trajectory
        )

        def _update_critic(agent_state, opt_state, sample_batch, key):
            """Apply one critic update and return new model and optimizer snapshots."""
            models = nnx.merge(agent_state.model_state, copy=True)

            def loss_fn(models, sample_batch, key):
                losses = self.agent.critic_loss(
                    agent_state.replace(model_state=nnx.unpack(models)),
                    sample_batch,
                    key,
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
            agent_state = agent_state.replace(model_state=nnx.unpack(models))
            return value, agent_state, nnx.unpack(optimizer)

        def _update_actor(agent_state, opt_state, sample_batch, key):
            """Apply one actor update and return new model and optimizer snapshots."""
            models = nnx.merge(agent_state.model_state, copy=True)

            def loss_fn(models, sample_batch, key):
                losses = self.agent.actor_loss(
                    agent_state.replace(model_state=nnx.unpack(models)),
                    sample_batch,
                    key,
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
            agent_state = agent_state.replace(model_state=nnx.unpack(models))
            return value, agent_state, nnx.unpack(optimizer)

        def _update_alpha(agent_state, opt_state, sample_batch, key):
            """Apply one alpha update and return new model and optimizer snapshots."""
            models = nnx.merge(agent_state.model_state, copy=True)

            def loss_fn(models, sample_batch, key):
                losses = self.agent.alpha_loss(
                    agent_state.replace(model_state=nnx.unpack(models)),
                    sample_batch,
                    key,
                )
                return losses.alpha_loss, losses

            value, grads = pmean_nnx_loss_and_grad(
                loss_fn,
                self.dp_axis_name,
                has_aux=True,
                wrt=nnx.All(nnx.Param, nnx.PathContains("log_alpha")),
            )(models, sample_batch, key)
            alpha = nnx.Dict(log_alpha=models.log_alpha)
            optimizer = nnx.merge(opt_state, copy=True)
            optimizer.update(alpha, grads)
            agent_state = agent_state.replace(model_state=nnx.unpack(models))
            return value, agent_state, nnx.unpack(optimizer)

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            critic_opt_state = opt_state.critic
            actor_opt_state = opt_state.actor

            key, critic_key, actor_key, alpha_key, rb_key = jax.random.split(key, num=5)

            if self.config.actor_update_interval - 1 > 0:

                def _sample_and_update_critic_fn(carry, unused_t):
                    key, agent_state, critic_opt_state = carry

                    key, rb_key, critic_key = jax.random.split(key, num=3)
                    # it's safe to use read-only replay_buffer_state here.
                    sample_batch = self.replay_buffer.sample(
                        replay_buffer_state, rb_key
                    )

                    (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
                        _update_critic(
                            agent_state,
                            critic_opt_state,
                            sample_batch,
                            critic_key,
                        )
                    )

                    return (key, agent_state, critic_opt_state), None

                key, critic_multiple_update_key = jax.random.split(key)

                (_, agent_state, critic_opt_state), _ = jax.lax.scan(
                    _sample_and_update_critic_fn,
                    (critic_multiple_update_key, agent_state, critic_opt_state),
                    (),
                    length=self.config.actor_update_interval - 1,
                )

            sample_batch = self.replay_buffer.sample(replay_buffer_state, rb_key)

            (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
                _update_critic(
                    agent_state,
                    critic_opt_state,
                    sample_batch,
                    critic_key,
                )
            )

            (actor_loss, actor_loss_dict), agent_state, actor_opt_state = _update_actor(
                agent_state,
                actor_opt_state,
                sample_batch,
                actor_key,
            )

            opt_state = opt_state.replace(
                actor=actor_opt_state, critic=critic_opt_state
            )

            if self.config.adaptive_alpha:
                # we follow the update order of the official implementation:
                # critic -> actor -> alpha
                alpha_opt_state = opt_state.alpha
                (alpha_loss, alpha_loss_dict), agent_state, alpha_opt_state = (
                    _update_alpha(
                        agent_state,
                        alpha_opt_state,
                        sample_batch,
                        alpha_key,
                    )
                )
                opt_state = opt_state.replace(alpha=alpha_opt_state)

                log_alpha = nnx.merge(agent_state.model_state, copy=True).log_alpha[...]
                alpha_loss_dict = alpha_loss_dict.replace(
                    log_alpha=log_alpha,
                    alpha=jnp.exp(log_alpha),
                )

                res = (
                    critic_loss,
                    actor_loss,
                    alpha_loss,
                    critic_loss_dict,
                    actor_loss_dict,
                    alpha_loss_dict,
                )
            else:
                res = (critic_loss, actor_loss, critic_loss_dict, actor_loss_dict)

            models = nnx.merge(agent_state.model_state, copy=True)
            nnx.update(
                models.target_critic,
                soft_target_update(
                    nnx.state(models.target_critic, nnx.Param),
                    nnx.state(models.critic, nnx.Param),
                    self.config.tau,
                ),
            )
            agent_state = agent_state.replace(model_state=nnx.unpack(models))

            return (key, agent_state, opt_state), res

        if self.config.adaptive_alpha:
            (
                (_, agent_state, opt_state),
                (
                    critic_loss,
                    actor_loss,
                    alpha_loss,
                    critic_loss_dict,
                    actor_loss_dict,
                    alpha_loss_dict,
                ),
            ) = scan_and_mean(
                _sample_and_update_fn,
                (learn_key, agent_state, state.opt_state),
                (),
                length=self.config.num_updates_per_iter,
            )
            train_metrics = SACTrainMetric(
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                alpha_loss=alpha_loss,
                raw_loss_dict=PyTreeDict(
                    {**critic_loss_dict, **actor_loss_dict, **alpha_loss_dict}
                ),
            ).all_reduce(axis_name=self.dp_axis_name)
        else:
            (
                (_, agent_state, opt_state),
                (
                    critic_loss,
                    actor_loss,
                    critic_loss_dict,
                    actor_loss_dict,
                ),
            ) = scan_and_mean(
                _sample_and_update_fn,
                (learn_key, agent_state, state.opt_state),
                (),
                length=self.config.num_updates_per_iter,
            )
            train_metrics = SACTrainMetric(
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
            ).all_reduce(axis_name=self.dp_axis_name)

        # calculate the number of timestep
        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.dp_axis_name,
        )

        sampled_epsiodes = psum(
            trajectory_dones.sum().astype(jnp.uint32), axis_name=self.dp_axis_name
        )

        # iterations is the number of updates of the agent
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_epsiodes,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(axis_name=self.dp_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            replay_buffer_state=replay_buffer_state,
            opt_state=opt_state,
        )
