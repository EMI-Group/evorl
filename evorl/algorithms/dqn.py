from evorl.utils import running_statistics
import logging
import math
from typing import Any

import chex
import distrax

from flax import nnx
import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig

from evorl.replay_buffers import ReplayBuffer
from evorl.distributed import psum, pmean
from evorl.distributed.gradients import pmean_nnx_loss_and_grad
from evorl.envs import AutoresetMode, Discrete, create_env, Space
from evorl.evaluators import Evaluator
from evorl.metrics import MetricBase, WorkflowMetric, metric_field
from evorl.networks import make_discrete_q_network
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
from evorl.utils.jax_utils import scan_and_mean, tree_stop_gradient, rng_fold_in_axis
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update

from evorl.agent import Agent, AgentState
from .offpolicy_utils import OffPolicyWorkflowTemplate, clean_trajectory

logger = logging.getLogger(__name__)


class DQNTrainMetric(MetricBase):
    # The primary loss is already averaged by pmean_nnx_loss_and_grad.
    loss: jax.Array = metric_field(default_factory=lambda: jnp.zeros(()))
    raw_loss_dict: LossDict = metric_field(default_factory=PyTreeDict, reduce_fn=pmean)


class DQNWorkflowMetric(WorkflowMetric):
    training_updates: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )  # not need sync


class DQNAgent(Agent):
    agent_network_cfg: DictConfig = pytree_field(static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)
    discount: float = 0.99
    target_type: str = "DDQN"

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        if not isinstance(action_space, Discrete):
            raise TypeError(f"Unsupported action space: {action_space}")

        cfg = self.agent_network_cfg
        value_obs_key = cfg.get("value_obs_key")
        q = make_discrete_q_network(
            input_size=(
                obs_space.spaces[value_obs_key] if value_obs_key else obs_space
            ).shape[-1],
            action_size=action_space.n,
            hidden_layer_sizes=tuple(cfg.q_hidden_layer_sizes),
            obs_key=value_obs_key,
            rngs=nnx.Rngs(key),
        )
        models = nnx.Dict(q=q, target_q=nnx.clone(q))

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
            extra_state=PyTreeDict(exploration_epsilon=jnp.zeros(())),
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        qs = self.get_models(agent_state, copy=True).q(obs)
        actions_dist = distrax.EpsilonGreedy(
            qs, epsilon=agent_state.extra_state.exploration_epsilon
        )
        # [B]: int from 0~(n-1)
        actions = actions_dist.sample(seed=key)

        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        qs = self.get_models(agent_state, copy=True).q(obs)

        actions_dist = distrax.EpsilonGreedy(
            qs, epsilon=agent_state.extra_state.exploration_epsilon
        )
        actions = actions_dist.mode()

        return actions, PyTreeDict()

    def loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        models = self.get_models(agent_state)
        obs = sample_batch.obs
        actions = sample_batch.actions
        rewards = sample_batch.rewards
        next_obs = sample_batch.extras.env_extras.ori_obs

        if self.normalize_obs:
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        discounts = self.discount * (1 - sample_batch.extras.env_extras.termination)

        qs = models.q(obs)
        # [B,n]->[B]
        qs = jnp.take_along_axis(qs, actions[..., None], axis=-1).squeeze(-1)

        # DQN_target: [B,n]
        next_qs = models.target_q(next_obs)

        if self.target_type == "DDQN":
            next_actions = models.q(next_obs).argmax(axis=-1, keepdims=True)  # [B,1]
            next_qs = jnp.take_along_axis(next_qs, next_actions, axis=-1).squeeze(-1)
        elif self.target_type == "DQN":
            next_qs = next_qs.max(axis=-1)  # [B,n]->[B]
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

        qs_target = jax.lax.stop_gradient(rewards + discounts * next_qs)

        q_loss = optax.squared_error(qs, qs_target).mean()

        return PyTreeDict(q_loss=q_loss, q_value=qs.mean())


class DQNWorkflow(OffPolicyWorkflowTemplate):
    @classmethod
    def name(cls):
        return "DQN"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )

        assert isinstance(env.action_space, Discrete), (
            "Only Discrete action space is supported."
        )

        agent = DQNAgent(
            agent_network_cfg=config.agent_network,
            discount=config.discount,
            target_type=config.target_type,
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

        return cls(env, agent, optimizer, evaluator, replay_buffer, config)

    def epsilon_scheduler(self, step: jax.Array | int) -> jax.Array:
        """Schedule exploration by global training updates, not per-device steps."""
        config = self.config
        num_iterations = (
            math.ceil(
                config.total_timesteps
                / (
                    config.num_envs
                    * self.num_devices
                    * config.rollout_length
                    * config.fold_iters
                )
            )
            * config.fold_iters
        )
        total_training_updates = num_iterations * config.num_updates_per_iter
        return optax.linear_schedule(
            init_value=config.exploration_epsilon.start,
            end_value=config.exploration_epsilon.end,
            transition_steps=(
                config.exploration_epsilon.exploration_fraction * total_training_updates
            )
            - 1,
        )(step)

    def _setup_workflow_metrics(self) -> MetricBase:
        return DQNWorkflowMetric()

    def _setup_agent_and_optimizer(
        self, key: jax.Array
    ) -> tuple[AgentState, chex.ArrayTree]:
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        models = nnx.merge(agent_state.model_state, copy=True)
        opt_state = nnx.unpack(nnx.Optimizer(models.q, self.optimizer, wrt=nnx.Param))

        agent_state = agent_state.replace(
            extra_state=agent_state.extra_state.replace(
                exploration_epsilon=self.epsilon_scheduler(0)
            )
        )

        return agent_state, opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key, buffer_key = jax.random.split(state.key, num=4)
        rollout_key, learn_key, buffer_key = rng_fold_in_axis(
            (rollout_key, learn_key, buffer_key), self.dp_axis_name
        )

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

        workflow_metrics = state.metrics

        def _update_q(agent_state, opt_state, sample_batch, key):
            """Apply one q update and return new model and optimizer snapshots."""
            models = nnx.merge(agent_state.model_state, copy=True)

            def loss_fn(models, sample_batch, key):
                losses = self.agent.loss(
                    agent_state.replace(model_state=nnx.unpack(models)),
                    sample_batch,
                    key,
                )
                return losses.q_loss, losses

            value, grads = pmean_nnx_loss_and_grad(
                loss_fn,
                self.dp_axis_name,
                has_aux=True,
                wrt=nnx.All(nnx.Param, nnx.PathContains("q")),
            )(models, sample_batch, key)
            optimizer = nnx.merge(opt_state, copy=True)
            optimizer.update(models.q, grads["q"])
            agent_state = agent_state.replace(model_state=nnx.unpack(models))
            return value, agent_state, nnx.unpack(optimizer)

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state, wf_metrics = carry

            key, rb_key, q_key = jax.random.split(key, 3)

            sample_batch = self.replay_buffer.sample(replay_buffer_state, rb_key)

            (q_loss, loss_dict), agent_state, opt_state = _update_q(
                agent_state,
                opt_state,
                sample_batch,
                q_key,
            )

            wf_metrics = wf_metrics.replace(
                training_updates=wf_metrics.training_updates + 1
            )

            def _soft_update_q(agent_state):
                models = nnx.merge(agent_state.model_state, copy=True)
                nnx.update(
                    models.target_q,
                    soft_target_update(
                        nnx.state(models.target_q, nnx.Param),
                        nnx.state(models.q, nnx.Param),
                        self.config.tau,
                    ),
                )
                return agent_state.replace(model_state=nnx.unpack(models))

            agent_state = jax.lax.cond(
                wf_metrics.training_updates % self.config.target_network_update_freq
                == 0,
                _soft_update_q,
                lambda agent_state: agent_state,
                agent_state,
            )

            agent_state = agent_state.replace(
                extra_state=agent_state.extra_state.replace(
                    exploration_epsilon=self.epsilon_scheduler(
                        wf_metrics.training_updates
                    )
                )
            )

            return (key, agent_state, opt_state, wf_metrics), (q_loss, loss_dict)

        (_, agent_state, opt_state, workflow_metrics), (q_loss, loss_dict) = (
            scan_and_mean(
                _sample_and_update_fn,
                (learn_key, agent_state, state.opt_state, state.metrics),
                (),
                length=self.config.num_updates_per_iter,
            )
        )

        train_metrics = DQNTrainMetric(
            loss=q_loss,
            raw_loss_dict=loss_dict,
        ).all_reduce(axis_name=self.dp_axis_name)

        # calculate the number of timestep
        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.dp_axis_name,
        )
        sampled_epsiodes = psum(
            trajectory_dones.sum().astype(jnp.uint32), axis_name=self.dp_axis_name
        )

        workflow_metrics = workflow_metrics.replace(
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
