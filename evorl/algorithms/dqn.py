import logging
import math
from typing import Any

import chex
import distrax

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from omegaconf import DictConfig

from evorl.replay_buffers import ReplayBuffer
from evorl.distributed import psum, pmean
from evorl.distributed.gradients import agent_gradient_update
from evorl.envs import AutoresetMode, Discrete, create_env, Space
from evorl.evaluators import Evaluator
from evorl.metrics import MetricBase, WorkflowMetric, metric_field
from evorl.networks import make_discrete_q_network
from evorl.rollout import rollout
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
from evorl.utils.jax_utils import scan_and_mean, tree_stop_gradient, tree_get
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update

from evorl.agent import Agent, AgentState
from .offpolicy_utils import OffPolicyWorkflowTemplate, clean_trajectory

logger = logging.getLogger(__name__)


class DQNNetworkParams(PyTreeData):
    q_params: Params
    target_q_params: Params
    exploration_epsilon: float


class DQNTrainMetric(MetricBase):
    # no need reduce_fn since it's already reduced in the step()
    loss: chex.Array = jnp.zeros(())
    raw_loss_dict: LossDict = metric_field(default_factory=PyTreeDict, reduce_fn=pmean)


class DQNWorkflowMetric(WorkflowMetric):
    training_updates: chex.Array = jnp.zeros((), dtype=jnp.uint32)  # not need sync


class DQNAgent(Agent):
    q_network: nn.Module
    obs_preprocessor: Any = pytree_field(default=None, static=True)
    discount: float = 0.99
    target_type: str = "DDQN"

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        dummy_obs = jtu.tree_map(lambda x: x[None, ...], obs_space.sample(key))

        q_params = self.q_network.init(key, dummy_obs)
        target_q_params = q_params

        params_states = DQNNetworkParams(
            q_params=q_params,
            target_q_params=target_q_params,
            exploration_epsilon=jnp.zeros(()),  # handle at workflow
        )

        if self.normalize_obs:
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(
                tree_get(dummy_obs, 0)
            )
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_states, obs_preprocessor_state=obs_preprocessor_state
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        qs = self.q_network.apply(agent_state.params.q_params, obs)
        # TODO: use tfp.Distribution
        actions_dist = distrax.EpsilonGreedy(
            qs, epsilon=agent_state.params.exploration_epsilon
        )
        # [B]: int from 0~(n-1)
        actions = actions_dist.sample(seed=key)

        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        qs = self.q_network.apply(agent_state.params.q_params, sample_batch.obs)

        actions_dist = distrax.EpsilonGreedy(
            qs, epsilon=agent_state.params.exploration_epsilon
        )
        actions = actions_dist.mode()

        return actions, PyTreeDict()

    def loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
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

        qs = self.q_network.apply(agent_state.params.q_params, obs)
        # [B,n]->[B]
        qs = jnp.take_along_axis(qs, actions[..., None], axis=-1).squeeze(-1)

        # DQN_target: [B,n]
        next_qs = self.q_network.apply(agent_state.params.target_q_params, next_obs)

        if self.target_type == "DDQN":
            next_actions = self.q_network.apply(
                agent_state.params.q_params, next_obs
            ).argmax(axis=-1, keepdims=True)  # [B,1]
            next_qs = jnp.take_along_axis(next_qs, next_actions, axis=-1).squeeze(-1)
        elif self.target_type == "DQN":
            next_qs = next_qs.max(axis=-1)  # [B,n]->[B]
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

        qs_target = jax.lax.stop_gradient(rewards + discounts * next_qs)

        q_loss = optax.squared_error(qs, qs_target).mean()

        return PyTreeDict(q_loss=q_loss, q_value=qs.mean())


def make_mlp_discrete_dqn_agent(
    action_space: Space,
    discount: float = 0.99,
    target_type: str = "DDQN",
    q_hidden_layer_sizes: tuple[int] = (256, 256),
    normalize_obs: bool = False,
    value_obs_key: str = "",
):
    assert isinstance(action_space, Discrete), (
        "Only Discrete action space is supported."
    )

    action_size = action_space.n
    q_network = make_discrete_q_network(
        action_size=action_size,
        hidden_layer_sizes=q_hidden_layer_sizes,
        obs_key=value_obs_key,
    )

    if normalize_obs:
        obs_preprocessor = running_statistics.normalize
    else:
        obs_preprocessor = None

    return DQNAgent(
        q_network=q_network,
        obs_preprocessor=obs_preprocessor,
        discount=discount,
        target_type=target_type,
    )


class DQNWorkflow(OffPolicyWorkflowTemplate):
    @classmethod
    def name(cls):
        return "DQN"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )

        assert isinstance(env.action_space, Discrete), (
            "Only Discrete action space is supported."
        )

        agent = make_mlp_discrete_dqn_agent(
            action_space=env.action_space,
            discount=config.discount,
            target_type=config.target_type,
            q_hidden_layer_sizes=config.agent_network.q_hidden_layer_sizes,
            normalize_obs=config.normalize_obs,
            value_obs_key=config.agent_network.value_obs_key,
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
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        workflow = cls(env, agent, optimizer, evaluator, replay_buffer, config)

        num_iterations = (
            math.ceil(
                config.total_timesteps
                / (config.num_envs * config.rollout_length * config.fold_iters)
            )
            * config.fold_iters
        )
        total_training_updates = num_iterations * config.num_updates_per_iter
        workflow.epsilon_scheduler = optax.linear_schedule(
            init_value=config.exploration_epsilon.start,
            end_value=config.exploration_epsilon.end,
            transition_steps=(
                config.exploration_epsilon.exploration_fraction * total_training_updates
            )
            - 1,
        )

        return workflow

    def _setup_workflow_metrics(self) -> MetricBase:
        return DQNWorkflowMetric()

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree]:
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        opt_state = self.optimizer.init(agent_state.params.q_params)

        agent_state = agent_state.replace(
            params=agent_state.params.replace(
                exploration_epsilon=self.epsilon_scheduler(0)
            )
        )

        return agent_state, opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key, buffer_key = jax.random.split(state.key, num=4)

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
                    pmap_axis_name=self.pmap_axis_name,
                )
            )

        replay_buffer_state = self.replay_buffer.add(
            state.replay_buffer_state, trajectory
        )

        def loss_fn(agent_state, sample_batch, key):
            loss_dict = self.agent.loss(agent_state, sample_batch, key)
            return loss_dict.q_loss, loss_dict

        q_update_fn = agent_gradient_update(
            loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True,
            attach_fn=lambda agent_state, q_params: agent_state.replace(
                params=agent_state.params.replace(q_params=q_params)
            ),
            detach_fn=lambda agent_state: agent_state.params.q_params,
        )

        workflow_metrics = state.metrics

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state, wf_metrics = carry

            key, rb_key, q_key = jax.random.split(key, 3)

            sample_batch = self.replay_buffer.sample(replay_buffer_state, rb_key)

            (q_loss, loss_dict), agent_state, opt_state = q_update_fn(
                opt_state, agent_state, sample_batch, q_key
            )

            wf_metrics = wf_metrics.replace(
                training_updates=wf_metrics.training_updates + 1
            )

            def _soft_update_q(agent_state):
                target_q_params = soft_target_update(
                    agent_state.params.target_q_params,
                    agent_state.params.q_params,
                    self.config.tau,
                )
                return agent_state.replace(
                    params=agent_state.params.replace(target_q_params=target_q_params)
                )

            agent_state = jax.lax.cond(
                wf_metrics.training_updates % self.config.target_network_update_freq
                == 0,
                _soft_update_q,
                lambda agent_state: agent_state,
                agent_state,
            )

            agent_state = agent_state.replace(
                params=agent_state.params.replace(
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
        )

        # calculate the number of timestep
        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.pmap_axis_name,
        )
        sampled_epsiodes = psum(
            trajectory_dones.sum().astype(jnp.uint32), axis_name=self.pmap_axis_name
        )

        workflow_metrics = workflow_metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_epsiodes,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            replay_buffer_state=replay_buffer_state,
            opt_state=opt_state,
        )
