import logging
import math
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from omegaconf import DictConfig

from evorl.replay_buffers import ReplayBuffer
from evorl.distributed import psum, unpmap
from evorl.distributed.gradients import gradient_update
from evorl.envs import AutoresetMode, Box, create_env, Space
from evorl.evaluators import Evaluator
from evorl.metrics import MetricBase
from evorl.rollout import rollout
from evorl.networks import make_policy_network, make_q_network
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action,
    Params,
    PyTreeData,
    PyTreeDict,
    PolicyExtraInfo,
    State,
    pytree_field,
)
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient, tree_get
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
        key, q_key1, q_key2, actor_key = jax.random.split(key, num=4)

        dummy_obs = jtu.tree_map(lambda x: x[None, ...], obs_space.sample(key))
        dummy_action = action_space.sample(key)[None, ...]

        critic1_params = self.critic_network.init(q_key1, dummy_obs, dummy_action)
        target_critic1_params = critic1_params
        critic2_params = self.critic_network.init(q_key2, dummy_obs, dummy_action)
        target_critic2_params = critic2_params

        actor_params = self.actor_network.init(actor_key, dummy_obs)
        target_actor_params = actor_params

        params_state = TD3NetworkParams(
            actor_params=actor_params,
            target_actor_params=target_actor_params,
            critic1_params=critic1_params,
            target_critic1_params=target_critic1_params,
            critic2_params=critic2_params,
            target_critic2_params=target_critic2_params,
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
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        actions = self.actor_network.apply(agent_state.params.actor_params, obs)

        return actions, PyTreeDict()


def make_mlp_td3_agent(
    action_space: Space,
    norm_layer_type: str = "none",
    critic_hidden_layer_sizes: tuple[int] = (256, 256),
    actor_hidden_layer_sizes: tuple[int] = (256, 256),
    discount: float = 0.99,
    exploration_epsilon: float = 0.5,
    policy_noise: float = 0.2,
    clip_policy_noise: float = 0.5,
    critics_in_actor_loss: str = "first",  #  or "min"
    normalize_obs: bool = False,
):
    assert isinstance(action_space, Box), "Only continue action space is supported."

    action_size = action_space.shape[0]

    critic_network = make_q_network(
        n_stack=1,
        hidden_layer_sizes=critic_hidden_layer_sizes,
        norm_layer_type=norm_layer_type,
    )
    actor_network = make_policy_network(
        action_size=action_size,
        hidden_layer_sizes=actor_hidden_layer_sizes,
        activation_final=nn.tanh,
        norm_layer_type=norm_layer_type,
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


class TD3TrainMetric(MetricBase):
    actor_loss: chex.Array
    critic1_loss: chex.Array
    critic2_loss: chex.Array
    q1: chex.Array
    q2: chex.Array


class TD3NetworkParams(PyTreeData):
    """Contains training state for the learner."""

    actor_params: Params
    critic1_params: Params
    critic2_params: Params
    target_actor_params: Params
    target_critic1_params: Params
    target_critic2_params: Params


class TD3V3Workflow(OffPolicyWorkflowTemplate):
    """The similar impl of TD3 in SB3 and CleanRL."""

    @classmethod
    def name(cls):
        return "TD3-V3"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )

        assert isinstance(env.action_space, Box), (
            "Only continue action space is supported."
        )

        agent = make_mlp_td3_agent(
            action_space=env.action_space,
            norm_layer_type=config.agent_network.norm_layer_type,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            critics_in_actor_loss=config.critics_in_actor_loss,
            normalize_obs=config.normalize_obs,
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
            critic1=self.optimizer.init(agent_state.params.critic1_params),
            critic2=self.optimizer.init(agent_state.params.critic2_params),
        )
        return agent_state, opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        iterations = state.metrics.iterations + 1
        key, rollout_key, critic_key, actor_key, rb_key = jax.random.split(
            state.key, num=5
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
                    pmap_axis_name=self.pmap_axis_name,
                )
            )

        replay_buffer_state = self.replay_buffer.add(
            state.replay_buffer_state, trajectory
        )

        def _update_critic_fn(agent_state, opt_state, sample_batch, key):
            critic1_opt_state = opt_state.critic1
            critic2_opt_state = opt_state.critic2

            agent = self.agent

            next_obs = sample_batch.extras.env_extras.ori_obs
            obs = sample_batch.obs
            actions = sample_batch.actions

            if agent.normalize_obs:
                next_obs = agent.obs_preprocessor(
                    next_obs, agent_state.obs_preprocessor_state
                )
                obs = agent.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

            actions_next = agent.actor_network.apply(
                agent_state.params.target_actor_params, next_obs
            )
            actions_next += jnp.clip(
                jax.random.normal(key, actions.shape) * agent.policy_noise,
                -agent.clip_policy_noise,
                agent.clip_policy_noise,
            )
            # Note: when calculating the critic loss, we also clip the actions to the action space
            actions_next = jnp.clip(actions_next, -1.0, 1.0)

            # [B]
            qs1_next = agent.critic_network.apply(
                agent_state.params.target_critic1_params, next_obs, actions_next
            )
            qs2_next = agent.critic_network.apply(
                agent_state.params.target_critic2_params, next_obs, actions_next
            )
            qs_next_min = jnp.minimum(qs1_next, qs2_next)

            discounts = agent.discount * (
                1 - sample_batch.extras.env_extras.termination
            )

            qs_target = sample_batch.rewards + discounts * qs_next_min
            qs_target = jax.lax.stop_gradient(qs_target)

            # q_loss = optax.huber_loss(qs, qs_target).sum(-1).mean()
            def _critic_loss(params):
                qs = agent.critic_network.apply(params, obs, actions)
                loss = optax.squared_error(qs, qs_target).mean()
                return loss, qs.mean()

            critic_update_fn = gradient_update(
                _critic_loss,
                self.optimizer,
                pmap_axis_name=self.pmap_axis_name,
                has_aux=True,
            )

            (critic1_loss, q1), ciritc1_params, critic1_opt_state = critic_update_fn(
                critic1_opt_state, agent_state.params.critic1_params
            )
            (critic2_loss, q2), ciritc2_params, critic2_opt_state = critic_update_fn(
                critic2_opt_state, agent_state.params.critic2_params
            )

            agent_state = agent_state.replace(
                params=agent_state.params.replace(
                    critic1_params=ciritc1_params, critic2_params=ciritc2_params
                )
            )

            opt_state = opt_state.replace(
                critic1=critic1_opt_state, critic2=critic2_opt_state
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
            actor_opt_state = opt_state.actor
            agent = self.agent
            obs = sample_batch.obs

            if agent.normalize_obs:
                obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

            # ==== actor update ====
            def _actor_loss(params):
                actions = agent.actor_network.apply(params, obs)
                loss = -agent.critic_network.apply(
                    agent_state.params.critic1_params, obs, actions
                ).mean()
                return loss

            actor_update_fn = gradient_update(
                _actor_loss,
                self.optimizer,
                pmap_axis_name=self.pmap_axis_name,
                has_aux=False,
            )

            actor_loss, actor_params, actor_opt_state = actor_update_fn(
                actor_opt_state, agent_state.params.actor_params
            )

            # soft update all target networks
            target_actor_params = soft_target_update(
                agent_state.params.target_actor_params,
                agent_state.params.actor_params,
                self.config.tau,
            )
            target_critic1_params = soft_target_update(
                agent_state.params.target_critic1_params,
                agent_state.params.critic1_params,
                self.config.tau,
            )
            target_critic2_params = soft_target_update(
                agent_state.params.target_critic2_params,
                agent_state.params.critic2_params,
                self.config.tau,
            )

            agent_state = agent_state.replace(
                params=agent_state.params.replace(
                    actor_params=actor_params,
                    target_actor_params=target_actor_params,
                    target_critic1_params=target_critic1_params,
                    target_critic2_params=target_critic2_params,
                )
            )

            opt_state = opt_state.replace(actor=actor_opt_state)

            train_info = PyTreeDict(
                actor_loss=actor_loss,
            )

            return (
                train_info,
                agent_state,
                opt_state,
            )

        def _dummy_update_actor_fn(agent_state, opt_state, sample_batch, key):
            return (
                PyTreeDict(actor_loss=MISSING_LOSS),
                agent_state,
                opt_state,
            )

        sample_batch = self.replay_buffer.sample(replay_buffer_state, rb_key)

        critic_train_info, agent_state, opt_state = _update_critic_fn(
            agent_state, opt_state, sample_batch, critic_key
        )

        # Note: using cond prohibits the parallel training with vmap
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
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        # calculate the number of timestep
        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.pmap_axis_name,
        )

        # iterations is the number of updates of the agent
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            iterations=iterations,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            replay_buffer_state=replay_buffer_state,
            opt_state=opt_state,
        )

    def learn(self, state: State) -> State:
        num_devices = jax.device_count()
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        sampled_timesteps = unpmap(state.metrics.sampled_timesteps).tolist()
        num_iters = math.ceil(
            (self.config.total_timesteps - sampled_timesteps)
            / (one_step_timesteps * self.config.fold_iters * num_devices)
        )
        start_iteration = unpmap(state.metrics.iterations, self.pmap_axis_name).tolist()
        final_iteration = num_iters + start_iteration

        for i in range(num_iters):
            train_metrics, state = self._multi_steps(state)
            workflow_metrics = state.metrics

            # current iteration
            iterations = unpmap(state.metrics.iterations, self.pmap_axis_name).tolist()
            train_metrics = unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = unpmap(workflow_metrics, self.pmap_axis_name)

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
                eval_metrics = unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iterations
                )

            saved_state = unpmap(state, self.pmap_axis_name)
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)
            self.checkpoint_manager.save(
                iterations,
                saved_state,
                force=iterations == final_iteration,
            )

        return state
