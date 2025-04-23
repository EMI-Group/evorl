import logging
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from omegaconf import DictConfig

from evorl.distributed import psum, pmean
from evorl.distributed.gradients import agent_gradient_update
from evorl.envs import AutoresetMode, Box, create_env, Space
from evorl.evaluators import Evaluator
from evorl.metrics import MetricBase, metric_field
from evorl.networks import make_policy_network, make_q_network
from evorl.rollout import rollout
from evorl.replay_buffers import ReplayBuffer
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


class TD3TrainMetric(MetricBase):
    critic_loss: chex.Array
    actor_loss: chex.Array
    raw_loss_dict: LossDict = metric_field(default_factory=PyTreeDict, reduce_fn=pmean)


class TD3NetworkParams(PyTreeData):
    """Contains training state for the learner."""

    actor_params: Params
    critic_params: Params
    target_actor_params: Params
    target_critic_params: Params


class TD3Agent(Agent):
    """The Agnet for TD3."""

    critic_network: nn.Module
    actor_network: nn.Module
    obs_preprocessor: Any = pytree_field(default=None, static=True)

    discount: float = 0.99
    exploration_epsilon: float = 0.5
    policy_noise: float = 0.2
    clip_policy_noise: float = 0.5
    critics_in_actor_loss: str = "first"  #  or "min"

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        key, q_key, actor_key = jax.random.split(key, num=3)

        dummy_obs = jtu.tree_map(lambda x: x[None, ...], obs_space.sample(key))
        dummy_action = action_space.sample(key)[None, ...]

        critic_params = self.critic_network.init(q_key, dummy_obs, dummy_action)
        target_critic_params = critic_params

        actor_params = self.actor_network.init(actor_key, dummy_obs)
        target_actor_params = actor_params

        params_state = TD3NetworkParams(
            critic_params=critic_params,
            actor_params=actor_params,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
        )

        if self.normalize_obs:
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(
                tree_get(dummy_obs, 0)
            )
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state,
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        # sample_barch: [#env, ...]

        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        actions = self.actor_network.apply(agent_state.params.actor_params, obs)
        # add random noise
        noise = jax.random.normal(key, actions.shape) * self.exploration_epsilon
        actions += noise
        actions = jnp.clip(actions, -1.0, 1.0)

        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        # sample_barch: [#env, ...]

        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        actions = self.actor_network.apply(agent_state.params.actor_params, obs)

        return actions, PyTreeDict()

    def critic_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        """Critic loss in TD3.

        Args:
            sample_barch: [B, ...]

        Return: LossDict[
            actor_loss
            critic_loss
            actor_entropy_loss
        ]
        """
        next_obs = sample_batch.extras.env_extras.ori_obs
        obs = sample_batch.obs
        actions = sample_batch.actions

        if self.normalize_obs:
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        next_actions = self.actor_network.apply(
            agent_state.params.target_actor_params, next_obs
        )
        next_actions += jnp.clip(
            jax.random.normal(key, actions.shape) * self.policy_noise,
            -self.clip_policy_noise,
            self.clip_policy_noise,
        )
        # Note: when calculating the critic loss, we also clip the actions to the action space
        next_actions = jnp.clip(next_actions, -1.0, 1.0)

        # [B, num_critics]
        next_qs = self.critic_network.apply(
            agent_state.params.target_critic_params, next_obs, next_actions
        )
        next_qs_min = next_qs.min(-1)

        discounts = self.discount * (1 - sample_batch.extras.env_extras.termination)

        qs_target = sample_batch.rewards + discounts * next_qs_min
        qs_target = jnp.broadcast_to(qs_target[..., None], (*qs_target.shape, 2))
        qs_target = jax.lax.stop_gradient(qs_target)

        qs = self.critic_network.apply(agent_state.params.critic_params, obs, actions)

        # q_loss = optax.huber_loss(qs, qs_target).sum(-1).mean()
        q_loss = optax.squared_error(qs, qs_target).sum(-1).mean()

        return PyTreeDict(critic_loss=q_loss, q_value=qs.mean())

    def actor_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        """Actor loss in TD3.

        Args:
            sample_barch: [B, ...]

        Return: LossDict[
            actor_loss
            critic_loss
            actor_entropy_loss
        ]
        """
        obs = sample_batch.obs

        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # [T*B, A]
        # Note: when calculating the actor loss, we don't clip the actions to the action space, following the impl of SB3 and CleanRL
        actions = self.actor_network.apply(agent_state.params.actor_params, obs)

        # TODO: handle redundant computation
        qs = self.critic_network.apply(agent_state.params.critic_params, obs, actions)

        if self.critics_in_actor_loss == "first":
            actor_loss = -jnp.mean(qs[..., 0])
        elif self.critics_in_actor_loss == "min":
            # using min_Q, like SAC
            actor_loss = -jnp.mean(qs.min(-1))
        else:
            raise ValueError(
                f"Invalid value for critics_in_actor_loss: {self.critics_in_actor_loss}, should be 'first' or 'mean'"
            )

        return PyTreeDict(actor_loss=actor_loss)


def make_mlp_td3_agent(
    action_space: Space,
    norm_layer_type: str = "none",
    num_critics: int = 2,
    critic_hidden_layer_sizes: tuple[int] = (256, 256),
    actor_hidden_layer_sizes: tuple[int] = (256, 256),
    discount: float = 0.99,
    exploration_epsilon: float = 0.5,
    policy_noise: float = 0.2,
    clip_policy_noise: float = 0.5,
    critics_in_actor_loss: str = "first",  #  or "min"
    normalize_obs: bool = False,
    policy_obs_key: str = "",
    value_obs_key: str = "",
):
    assert isinstance(action_space, Box), "Only continue action space is supported."

    action_size = action_space.shape[0]

    critic_network = make_q_network(
        n_stack=num_critics,
        hidden_layer_sizes=critic_hidden_layer_sizes,
        norm_layer_type=norm_layer_type,
        obs_key=value_obs_key,
    )
    actor_network = make_policy_network(
        action_size=action_size,
        hidden_layer_sizes=actor_hidden_layer_sizes,
        activation_final=nn.tanh,
        norm_layer_type=norm_layer_type,
        obs_key=policy_obs_key,
    )

    if normalize_obs:
        obs_preprocessor = running_statistics.normalize
    else:
        obs_preprocessor = None

    return TD3Agent(
        critic_network=critic_network,
        actor_network=actor_network,
        obs_preprocessor=obs_preprocessor,
        discount=discount,
        exploration_epsilon=exploration_epsilon,
        policy_noise=policy_noise,
        clip_policy_noise=clip_policy_noise,
        critics_in_actor_loss=critics_in_actor_loss,
    )


class TD3Workflow(OffPolicyWorkflowTemplate):
    @classmethod
    def name(cls):
        return "TD3"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )

        agent = make_mlp_td3_agent(
            action_space=env.action_space,
            norm_layer_type=config.agent_network.norm_layer_type,
            num_critics=config.agent_network.num_critics,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            critics_in_actor_loss=config.critics_in_actor_loss,
            normalize_obs=config.normalize_obs,
            policy_obs_key=config.agent_network.policy_obs_key,
            value_obs_key=config.agent_network.value_obs_key,
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
            episode_length=config.env.max_episode_steps,
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
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree]:
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        opt_state = PyTreeDict(
            actor=self.optimizer.init(agent_state.params.actor_params),
            critic=self.optimizer.init(agent_state.params.critic_params),
        )
        return agent_state, opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

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

        def critic_loss_fn(agent_state, sample_batch, key):
            loss_dict = self.agent.critic_loss(agent_state, sample_batch, key)

            loss = loss_dict.critic_loss
            return loss, loss_dict

        def actor_loss_fn(agent_state, sample_batch, key):
            loss_dict = self.agent.actor_loss(agent_state, sample_batch, key)

            loss = loss_dict.actor_loss
            return loss, loss_dict

        critic_update_fn = agent_gradient_update(
            critic_loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True,
            attach_fn=lambda agent_state, critic_params: agent_state.replace(
                params=agent_state.params.replace(critic_params=critic_params)
            ),
            detach_fn=lambda agent_state: agent_state.params.critic_params,
        )

        actor_update_fn = agent_gradient_update(
            actor_loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True,
            attach_fn=lambda agent_state, actor_params: agent_state.replace(
                params=agent_state.params.replace(actor_params=actor_params)
            ),
            detach_fn=lambda agent_state: agent_state.params.actor_params,
        )

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
                        critic_update_fn(
                            critic_opt_state, agent_state, sample_batch, critic_key
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
                critic_update_fn(
                    critic_opt_state, agent_state, sample_batch, critic_key
                )
            )

            (actor_loss, actor_loss_dict), agent_state, actor_opt_state = (
                actor_update_fn(actor_opt_state, agent_state, sample_batch, actor_key)
            )

            target_actor_params = soft_target_update(
                agent_state.params.target_actor_params,
                agent_state.params.actor_params,
                self.config.tau,
            )
            target_critic_params = soft_target_update(
                agent_state.params.target_critic_params,
                agent_state.params.critic_params,
                self.config.tau,
            )
            agent_state = agent_state.replace(
                params=agent_state.params.replace(
                    target_actor_params=target_actor_params,
                    target_critic_params=target_critic_params,
                )
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
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        # calculate the number of timestep
        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.pmap_axis_name,
        )

        sampled_epsiodes = psum(
            trajectory_dones.sum().astype(jnp.uint32), axis_name=self.pmap_axis_name
        )

        # iterations is the number of updates of the agent
        workflow_metrics = state.metrics.replace(
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
