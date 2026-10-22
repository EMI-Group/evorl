from evorl.utils import running_statistics
import logging
from typing import Any

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig

from evorl.distributed import psum, pmean
from evorl.distributed.gradients import pmean_nnx_loss_and_grad
from evorl.envs import Box, AutoresetMode, create_env, Space
from evorl.evaluators import Evaluator
from evorl.metrics import MetricBase, metric_field
from evorl.networks import make_policy_network, make_q_network
from evorl.rollout import rollout
from evorl.replay_buffers import ReplayBuffer
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


class TD3TrainMetric(MetricBase):
    # Primary losses are already averaged by pmean_nnx_loss_and_grad.
    critic_loss: jax.Array
    actor_loss: jax.Array
    raw_loss_dict: LossDict = metric_field(default_factory=PyTreeDict, reduce_fn=pmean)


class TD3Agent(Agent):
    """The Agent for TD3."""

    agent_network_cfg: DictConfig = pytree_field(static=True)

    obs_preprocessor: Any = pytree_field(default=None, static=True)

    discount: float = 0.99
    exploration_epsilon: float = 0.5
    policy_noise: float = 0.2
    clip_policy_noise: float = 0.5
    critics_in_actor_loss: str = "first"  #  or "min"

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        if not isinstance(action_space, Box):
            raise TypeError(f"Unsupported action space: {action_space}")
        if self.agent_network_cfg.get("num_critics", 2) < 2:
            raise ValueError("num_critics must be at least 2 for twin-critic training")
        key, q_key, actor_key = jax.random.split(key, num=3)

        cfg = self.agent_network_cfg
        value_obs_key = cfg.get("value_obs_key")
        policy_obs_key = cfg.get("policy_obs_key")
        critic = make_q_network(
            input_size=(
                obs_space.spaces[value_obs_key] if value_obs_key else obs_space
            ).shape[-1],
            action_size=action_space.shape[0],
            hidden_layer_sizes=tuple(cfg.critic_hidden_layer_sizes),
            obs_key=value_obs_key,
            norm_layer_type=cfg.get("norm_layer_type", "none"),
            n_stack=cfg.num_critics,
            rngs=nnx.Rngs(q_key),
        )
        actor = make_policy_network(
            input_size=(
                obs_space.spaces[policy_obs_key] if policy_obs_key else obs_space
            ).shape[-1],
            action_size=action_space.shape[0],
            hidden_layer_sizes=tuple(cfg.actor_hidden_layer_sizes),
            activation_final=jax.nn.tanh,
            obs_key=policy_obs_key,
            norm_layer_type=cfg.get("norm_layer_type", "none"),
            rngs=nnx.Rngs(actor_key),
        )
        models = nnx.Dict(
            actor=actor,
            critic=critic,
            target_actor=nnx.clone(actor),
            target_critic=nnx.clone(critic),
        )

        if self.normalize_obs:
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(
                obs_space.sample(key)
            )
        else:
            obs_preprocessor_state = None

        return AgentState(
            model_state=nnx.unpack(models),
            obs_preprocessor_state=obs_preprocessor_state,
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        # sample_barch: [#env, ...]

        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        actions = self.get_models(agent_state, copy=True).actor(obs)
        # add random noise
        noise = jax.random.normal(key, actions.shape) * self.exploration_epsilon
        actions += noise
        actions = jnp.clip(actions, -1.0, 1.0)

        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        # sample_barch: [#env, ...]

        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        actions = self.get_models(agent_state, copy=True).actor(obs)

        return actions, PyTreeDict()

    def critic_loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        """Compute the TD3 critic loss from a working agent state.

        Args:
            agent_state: Training state sharing variables with the working models.
            sample_batch: Batch of transitions with leading sample dimension.
            key: Random key for loss computation.

        Returns:
            A dictionary containing the critic loss and mean Q value.
        """
        models = self.get_models(agent_state)
        next_obs = sample_batch.extras.env_extras.ori_obs
        obs = sample_batch.obs
        actions = sample_batch.actions

        if self.normalize_obs:
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        next_actions = models.target_actor(next_obs)
        next_actions += jnp.clip(
            jax.random.normal(key, actions.shape) * self.policy_noise,
            -self.clip_policy_noise,
            self.clip_policy_noise,
        )
        # Note: when calculating the critic loss, we also clip the actions to the action space
        next_actions = jnp.clip(next_actions, -1.0, 1.0)

        # [B, num_critics]
        next_qs = models.target_critic(next_obs, next_actions)
        next_qs_min = next_qs.min(-1)

        discounts = self.discount * (1 - sample_batch.extras.env_extras.termination)

        qs_target = sample_batch.rewards + discounts * next_qs_min
        qs = models.critic(obs, actions)
        qs_target = jax.lax.stop_gradient(
            jnp.broadcast_to(qs_target[..., None], qs.shape)
        )

        # q_loss = optax.huber_loss(qs, qs_target).sum(-1).mean()
        q_loss = optax.squared_error(qs, qs_target).sum(-1).mean()

        return PyTreeDict(critic_loss=q_loss, q_value=qs.mean())

    def actor_loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        """Compute the TD3 actor loss from a working agent state.

        Args:
            agent_state: Training state sharing variables with the working models.
            sample_batch: Batch of transitions with leading sample dimension.
            key: Random key for loss computation.

        Returns:
            A dictionary containing the actor loss.
        """
        models = self.get_models(agent_state)
        obs = sample_batch.obs

        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # [T*B, A]
        # Note: when calculating the actor loss, we don't clip the actions to the action space, following the impl of SB3 and CleanRL
        actions = models.actor(obs)

        # TODO: handle redundant computation
        qs = models.critic(obs, actions)

        if self.critics_in_actor_loss == "first":
            actor_loss = -jnp.mean(qs[..., 0])
        elif self.critics_in_actor_loss == "min":
            # using min_Q, like SAC
            actor_loss = -jnp.mean(qs.min(-1))
        else:
            raise ValueError(
                f"Invalid value for critics_in_actor_loss: {self.critics_in_actor_loss}, should be 'first' or 'min'"
            )

        return PyTreeDict(actor_loss=actor_loss)


class TD3Workflow(OffPolicyWorkflowTemplate):
    @classmethod
    def name(cls):
        return "TD3"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )

        agent = TD3Agent(
            agent_network_cfg=config.agent_network,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            critics_in_actor_loss=config.critics_in_actor_loss,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
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
            actor=nnx.unpack(
                nnx.Optimizer(models.actor, self.optimizer, wrt=nnx.Param)
            ),
            critic=nnx.unpack(
                nnx.Optimizer(models.critic, self.optimizer, wrt=nnx.Param)
            ),
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
            critic_state = nnx.clone(nnx.state(models.critic, nnx.Not(nnx.Param)))

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
            # The actor update only retains the actor's auxiliary variables.
            nnx.update(models.critic, critic_state)
            agent_state = agent_state.replace(model_state=nnx.unpack(models))
            return value, agent_state, nnx.unpack(optimizer)

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            critic_opt_state = opt_state.critic
            actor_opt_state = opt_state.actor

            key, critic_key, actor_key, rb_key = jax.random.split(key, num=4)

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

            models = nnx.merge(agent_state.model_state, copy=True)
            nnx.update(
                models.target_actor,
                soft_target_update(
                    nnx.state(models.target_actor, nnx.Param),
                    nnx.state(models.actor, nnx.Param),
                    self.config.tau,
                ),
            )
            nnx.update(
                models.target_critic,
                soft_target_update(
                    nnx.state(models.target_critic, nnx.Param),
                    nnx.state(models.critic, nnx.Param),
                    self.config.tau,
                ),
            )
            agent_state = agent_state.replace(
                model_state=nnx.unpack(models),
            )

            opt_state = opt_state.replace(
                actor=actor_opt_state, critic=critic_opt_state
            )

            return (
                (key, agent_state, opt_state),
                (critic_loss, actor_loss, critic_loss_dict, actor_loss_dict),
            )

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

        train_metrics = TD3TrainMetric(
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
