from evorl.utils import running_statistics
import logging
import math
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from omegaconf import DictConfig

from evorl.replay_buffers import ReplayBuffer
from evorl.distributed import pmean, psum
from evorl.distributed.gradients import pmean_nnx_loss_and_grad
from evorl.distributed.sharding import mark_varying
from evorl.envs import AutoresetMode, Box, create_env, Space
from evorl.evaluators import Evaluator
from evorl.metrics import MetricBase, metric_field
from evorl.rollout import rollout
from evorl.networks import make_policy_network, make_q_network
from evorl.types import (
    PyTreeDict,
    State,
    pytree_field,
)
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update
from evorl.agent import AgentState, Agent
from evorl.recorders import add_prefix

from ..offpolicy_utils import (
    OffPolicyWorkflowTemplate,
    clean_trajectory,
    skip_replay_buffer_state,
)


logger = logging.getLogger(__name__)

MISSING_LOSS = -1e10


class TD3Agent(Agent):
    """TD3 with two separately optimized critics."""

    agent_network_cfg: DictConfig = pytree_field(static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)
    discount: float = 0.99
    exploration_epsilon: float = 0.5
    policy_noise: float = 0.2
    clip_policy_noise: float = 0.5
    critics_in_actor_loss: str = "first"

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        if not isinstance(action_space, Box):
            raise TypeError(f"Unsupported action space: {action_space}")
        key, q_key1, q_key2, actor_key = jax.random.split(key, 4)
        cfg = self.agent_network_cfg
        actor = make_policy_network(
            input_size=obs_space.shape[-1],
            action_size=action_space.shape[0],
            hidden_layer_sizes=cfg.actor_hidden_layer_sizes,
            activation_final=jnp.tanh,
            norm_layer_type=cfg.get("norm_layer_type", "none"),
            rngs=nnx.Rngs(actor_key),
        )

        def create_critic(key):
            return make_q_network(
                input_size=obs_space.shape[-1],
                action_size=action_space.shape[0],
                n_stack=1,
                hidden_layer_sizes=cfg.critic_hidden_layer_sizes,
                norm_layer_type=cfg.get("norm_layer_type", "none"),
                rngs=nnx.Rngs(key),
            )

        critic1, critic2 = create_critic(q_key1), create_critic(q_key2)
        models = nnx.Dict(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            target_actor=nnx.clone(actor, arrays=True),
            target_critic1=nnx.clone(critic1, arrays=True),
            target_critic2=nnx.clone(critic2, arrays=True),
        )
        obs_state = (
            running_statistics.init_state(obs_space.sample(key))
            if self.normalize_obs
            else None
        )
        return AgentState(
            model_state=nnx.unpack(models), obs_preprocessor_state=obs_state
        )

    def compute_actions(self, agent_state, sample_batch, key):
        actions, _ = self.evaluate_actions(agent_state, sample_batch, key)
        noise = jax.random.normal(key, actions.shape) * self.exploration_epsilon
        return jnp.clip(actions + noise, -1.0, 1.0), PyTreeDict()

    def evaluate_actions(self, agent_state, sample_batch, key):
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        models = nnx.merge(agent_state.model_state, copy=True)
        return models.actor(obs), PyTreeDict()


class TD3TrainMetric(MetricBase):
    # Primary losses are already averaged by pmean_loss_and_grad.
    actor_loss: jax.Array
    critic1_loss: jax.Array
    critic2_loss: jax.Array
    q1: jax.Array = metric_field(reduce_fn=pmean)
    q2: jax.Array = metric_field(reduce_fn=pmean)


class TD3V3Workflow(OffPolicyWorkflowTemplate):
    """The similar impl of TD3 in SB3 and CleanRL."""

    @classmethod
    def name(cls):
        return "TD3-V3"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )

        assert isinstance(env.action_space, Box), (
            "Only continue action space is supported."
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

    def _setup_agent_and_optimizer(self, key):
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        models = nnx.merge(agent_state.model_state, copy=True)
        opt_state = PyTreeDict(
            actor=nnx.unpack(
                nnx.Optimizer(models.actor, self.optimizer, wrt=nnx.Param)
            ),
            critic1=nnx.unpack(
                nnx.Optimizer(models.critic1, self.optimizer, wrt=nnx.Param)
            ),
            critic2=nnx.unpack(
                nnx.Optimizer(models.critic2, self.optimizer, wrt=nnx.Param)
            ),
        )
        return agent_state, opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        iterations = state.metrics.iterations + 1
        key, rollout_key, critic_key, actor_key, rb_key = jax.random.split(
            state.key, num=5
        )
        if self.is_multi_device:
            rank = jax.lax.axis_index(self.dp_axis_name)
            rollout_key = jax.random.fold_in(rollout_key, rank)
            critic_key = jax.random.fold_in(critic_key, rank)
            actor_key = jax.random.fold_in(actor_key, rank)
            rb_key = jax.random.fold_in(rb_key, rank)

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

        trajectory = clean_trajectory(trajectory)
        trajectory = flatten_rollout_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        agent_state = state.agent_state
        opt_state = state.opt_state

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

        def _update_critic_fn(agent_state, opt_state, sample_batch, key):
            agent = self.agent
            models = nnx.merge(agent_state.model_state, copy=True)
            next_obs = sample_batch.extras.env_extras.ori_obs
            obs = sample_batch.obs
            if agent.normalize_obs:
                next_obs = agent.obs_preprocessor(
                    next_obs, agent_state.obs_preprocessor_state
                )
                obs = agent.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
            actions_next = models.target_actor(next_obs)
            actions_next += jnp.clip(
                jax.random.normal(key, sample_batch.actions.shape) * agent.policy_noise,
                -agent.clip_policy_noise,
                agent.clip_policy_noise,
            )
            actions_next = jnp.clip(actions_next, -1.0, 1.0)
            qs_next = jnp.minimum(
                models.target_critic1(next_obs, actions_next),
                models.target_critic2(next_obs, actions_next),
            )
            discounts = agent.discount * (
                1 - sample_batch.extras.env_extras.termination
            )
            qs_target = jax.lax.stop_gradient(
                sample_batch.rewards + discounts * qs_next
            )

            def critic_loss(critic):
                qs = critic(obs, sample_batch.actions)
                return optax.squared_error(qs, qs_target).mean(), qs.mean()

            loss_and_grad = pmean_nnx_loss_and_grad(
                critic_loss, self.dp_axis_name, has_aux=True
            )
            (critic1_loss, q1), grads1 = loss_and_grad(models.critic1)
            (critic2_loss, q2), grads2 = loss_and_grad(models.critic2)
            optimizer1 = nnx.merge(opt_state.critic1, copy=True)
            optimizer2 = nnx.merge(opt_state.critic2, copy=True)
            optimizer1.update(models.critic1, grads1)
            optimizer2.update(models.critic2, grads2)
            agent_state = agent_state.replace(model_state=nnx.unpack(models))
            opt_state = opt_state.replace(
                critic1=nnx.unpack(optimizer1), critic2=nnx.unpack(optimizer2)
            )

            train_info = PyTreeDict(
                critic1_loss=critic1_loss,
                critic2_loss=critic2_loss,
                q1=q1,
                q2=q2,
            )

            return (
                train_info,
                agent_state,
                opt_state,
            )

        def _update_actor_fn(agent_state, opt_state, sample_batch, key):
            agent = self.agent
            models = nnx.merge(agent_state.model_state, copy=True)
            obs = sample_batch.obs
            if agent.normalize_obs:
                obs = agent.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

            def actor_loss(models):
                return -models.critic1(obs, models.actor(obs)).mean()

            actor_loss, grads = pmean_nnx_loss_and_grad(
                actor_loss,
                self.dp_axis_name,
                wrt=nnx.All(nnx.Param, nnx.PathContains("actor")),
            )(models)
            optimizer = nnx.merge(opt_state.actor, copy=True)
            optimizer.update(models.actor, grads["actor"])
            for target, online in (
                (models.target_actor, models.actor),
                (models.target_critic1, models.critic1),
                (models.target_critic2, models.critic2),
            ):
                nnx.update(
                    target,
                    soft_target_update(
                        nnx.state(target, nnx.Param),
                        nnx.state(online, nnx.Param),
                        self.config.tau,
                    ),
                )
            agent_state = agent_state.replace(model_state=nnx.unpack(models))
            opt_state = opt_state.replace(actor=nnx.unpack(optimizer))

            train_info = PyTreeDict(
                actor_loss=actor_loss,
            )

            return (
                train_info,
                agent_state,
                opt_state,
            )

        def _dummy_update_actor_fn(agent_state, opt_state, sample_batch, key):
            actor_loss = jnp.asarray(MISSING_LOSS)
            if not self.is_multi_device:
                actor_loss = mark_varying(
                    actor_loss, jax.sharding.get_abstract_mesh().manual_axes
                )
            return (
                PyTreeDict(actor_loss=actor_loss),
                agent_state,
                opt_state,
            )

        sample_batch = self.replay_buffer.sample(replay_buffer_state, rb_key)

        critic_train_info, agent_state, opt_state = _update_critic_fn(
            agent_state, opt_state, sample_batch, critic_key
        )

        (
            actor_train_info,
            agent_state,
            opt_state,
        ) = jax.lax.cond(
            iterations % self.config.actor_update_interval == 0,
            _update_actor_fn,
            _dummy_update_actor_fn,
            agent_state,
            opt_state,
            sample_batch,
            actor_key,
        )

        train_metrics = TD3TrainMetric(
            **actor_train_info,
            **critic_train_info,
        ).all_reduce(axis_name=self.dp_axis_name)

        # calculate the number of timestep
        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.dp_axis_name,
        )

        # iterations is the number of updates of the agent
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            iterations=iterations,
        ).all_reduce(axis_name=self.dp_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            replay_buffer_state=replay_buffer_state,
            opt_state=opt_state,
        )

    def learn(self, state: State) -> State:
        num_devices = self.num_devices
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        sampled_timesteps = state.metrics.sampled_timesteps.tolist()
        num_fold_iters = math.ceil(
            (self.config.total_timesteps - sampled_timesteps)
            / (one_step_timesteps * self.config.fold_iters * num_devices)
        )
        start_iteration = state.metrics.iterations.tolist()
        final_iteration = start_iteration + num_fold_iters * self.config.fold_iters

        for i in range(num_fold_iters):
            train_metrics, state = self._multi_steps(state)
            workflow_metrics = state.metrics

            # current iteration
            iterations = state.metrics.iterations.tolist()

            train_metrics = jtu.tree_map(
                lambda x: None if x == MISSING_LOSS else x, train_metrics
            )

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
                iterations,
                saved_state,
                force=iterations == final_iteration,
            )

        return state
