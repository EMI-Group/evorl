import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import struct
import math
from functools import partial

from omegaconf import DictConfig


from evorl.sample_batch import SampleBatch
from evorl.networks import make_policy_network, make_value_network
from evorl.utils import running_statistics
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.utils.jax_utils import tree_stop_gradient, rng_split
from evorl.utils.toolkits import (
    compute_gae, flatten_rollout_trajectory,
    average_episode_discount_return, shuffle_sample_batch
)
from evorl.rollout import rollout
from evorl.workflows import OnPolicyRLWorkflow
from evorl.agents import AgentState
from evorl.distributed import agent_gradient_update, tree_unpmap, psum
from evorl.envs import create_env, Env, EnvState
from evorl.evaluator import Evaluator
from .agent import Agent, AgentState

from evox import State

import orbax.checkpoint as ocp
import chex
import optax
from evorl.types import (
    LossDict, Action, Params, PolicyExtraInfo, PyTreeDict, pytree_field,
    MISSING_REWARD, PyTreeData
)
from evorl.metrics import TrainMetric, WorkflowMetric
from typing import Any
import logging
import flax.linen as nn


logger = logging.getLogger(__name__)


class IMPALANetworkParams(PyTreeData):
    """Contains training state for the learner."""
    policy_params: Params
    value_params: Params


class IMPALATrainMetric(TrainMetric):
    rho: chex.Array = jnp.zeros((), dtype=jnp.float32)


class IMPALAAgent(Agent):
    actor_hidden_layer_sizes: tuple[int] = (256, 256)
    critic_hidden_layer_sizes: tuple[int] = (256, 256)
    normalize_obs: bool = False
    continuous_action: bool = False
    discount: float = 0.99
    vtrace_lambda: float = 1.0
    clip_rho_threshold: float = 1.0
    clip_c_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
    clip_ppo_epsilon: float = 0.2
    adv_mode: str = pytree_field(default='official', pytree_node=False)
    pg_loss_mode: str = pytree_field(default='a2c', pytree_node=False)
    policy_network: nn.Module = pytree_field(lazy_init=True)
    value_network: nn.Module = pytree_field(lazy_init=True)
    obs_preprocessor: Any = pytree_field(lazy_init=True, pytree_node=False)

    def init(self, key: chex.PRNGKey) -> AgentState:

        obs_size = self.obs_space.shape[0]

        if self.continuous_action:
            action_size = self.action_space.shape[0]
            action_size *= 2
        else:
            action_size = self.action_space.n

        policy_key, value_key, obs_preprocessor_key = jax.random.split(key, 3)
        policy_network, policy_init_fn = make_policy_network(
            action_size=action_size,
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes
        )
        policy_params = policy_init_fn(policy_key)

        value_network, value_init_fn = make_value_network(
            obs_size=obs_size,
            hidden_layer_sizes=self.critic_hidden_layer_sizes
        )
        value_params = value_init_fn(value_key)

        self.set_frozen_attr('policy_network', policy_network)
        self.set_frozen_attr('value_network', value_network)

        params_state = IMPALANetworkParams(
            policy_params=policy_params,
            value_params=value_params
        )

        if self.normalize_obs:
            obs_preprocessor = running_statistics.normalize
            self.set_frozen_attr('obs_preprocessor', obs_preprocessor)
            dummy_obs = self.obs_space.sample(obs_preprocessor_key)
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state
        )

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> tuple[Action, PolicyExtraInfo]:
        """
            Args:
                sample_barch: [#env, ...]
        """
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(
            agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(
                *jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.sample(seed=key)

        policy_extras = PyTreeDict(
            # Log probabilities of the selected actions for importance sampling
            logp=actions_dist.log_prob(actions)
            # raw_action=raw_actions,
        )

        return jax.lax.stop_gradient(actions), policy_extras

    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> tuple[Action, PolicyExtraInfo]:
        """
            Args:
                sample_barch: [#env, ...]
        """
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(
            agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(
                *jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.mode()

        return jax.lax.stop_gradient(actions), PyTreeDict()

    def loss(self, agent_state: AgentState, trajectory: SampleBatch, key: chex.PRNGKey) -> LossDict:
        """
            Args:
                trajectory: [T, B, ...]
                    a sequence of transitions, not shuffled timesteps

            Return: 
                LossDict[
                    actor_loss
                    critic_loss
                    actor_entropy_loss
                ]
        """

        obs = trajectory.obs
        last_obs = trajectory.extras.env_extras.last_obs
        _obs = jnp.concatenate([obs, last_obs[-1:]], axis=0)

        if self.normalize_obs:
            _obs = self.obs_preprocessor(
                _obs, agent_state.obs_preprocessor_state)

        vs = self.value_network.apply(
            agent_state.params.value_params, _obs).squeeze(-1)

        sampled_actions_logp = trajectory.extras.policy_extras.logp
        sampled_actions = trajectory.actions

        # [T, B, A]
        raw_actions = self.policy_network.apply(
            agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(
                *jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        # [T, B]
        actions_logp = actions_dist.log_prob(sampled_actions)
        rho = jnp.exp(actions_logp - sampled_actions_logp)

        vtrace = compute_vtrace(
            rho_t=rho,
            v_t=vs[:-1],
            v_t_plus_1=vs[1:],
            rewards=trajectory.rewards,
            dones=trajectory.dones,
            discount=self.discount,
            lambda_=self.vtrace_lambda,
            clip_rho_threshold=self.clip_rho_threshold,
            clip_c_threshold=self.clip_c_threshold
        )

        # ======= critic =======

        critic_loss = optax.l2_loss(vs[:-1], vtrace).mean()

        # ====== actor =======

        advantages = compute_pg_advantage(
            rho_t=rho,
            vtrace=vtrace,
            v_t=vs[:-1],
            v_t_plus_1=vs[1:],
            rewards=trajectory.rewards,
            dones=trajectory.dones,
            discount=self.discount,
            lambda_=self.vtrace_lambda,
            clip_pg_rho_threshold=self.clip_pg_rho_threshold,
            mode=self.adv_mode
        )
        advantages = jax.lax.stop_gradient(advantages)

        # advantages: [T*B]
        if self.pg_loss_mode == 'a2c':
            policy_loss = - (advantages * actions_logp).mean()
        elif self.pg_loss_mode == 'ppo':
            policy_sorrogate_loss1 = rho * advantages
            policy_sorrogate_loss2 = jnp.clip(
                rho, 1-self.clipping_epsilon, 1+self.clip_ppo_epsilon) * advantages
            policy_loss = - jnp.minimum(
                policy_sorrogate_loss1, policy_sorrogate_loss2).mean()
        else:
            raise ValueError(
                f'pg_loss_mode {self.pg_loss_mode} is not supported')

        # entropy: [T*B]
        if self.continuous_action:
            entropy_loss = actions_dist.entropy(seed=key).mean()
        else:
            entropy_loss = actions_dist.entropy().mean()

        return PyTreeDict(
            actor_loss=policy_loss,
            critic_loss=critic_loss,
            actor_entropy_loss=entropy_loss,
            rho=rho.mean()
        )


class IMPALAWorkflow(OnPolicyRLWorkflow):
    """
        Syncrhonous version of IMPALA (A2C|PPO w/ V-Trace)
    """
    @classmethod
    def name(cls):
        return "IMPALA"

    @staticmethod
    def _rescale_config(config: DictConfig, devices) -> None:
        num_devices = len(devices)
        if config.num_envs % num_devices != 0:
            logger.warning(
                f"num_envs({config.num_envs}) cannot be divided by num_devices({num_devices}), "
                f"rescale num_envs to {config.num_envs // num_devices}"
            )
        if config.num_eval_envs % num_devices != 0:
            logger.warning(
                f"num_eval_envs({config.num_eval_envs}) cannot be divided by num_devices({num_devices}), "
                f"rescale num_eval_envs to {config.num_eval_envs // num_devices}"
            )
        if config.minibatch_size % num_devices != 0:
            logger.warning(
                f"minibatch_size({config.minibatch_size}) cannot be divided by num_devices({num_devices}), "
                f"rescale minibatch_size to {config.minibatch_size//num_devices}"
            )

        config.num_envs = config.num_envs // num_devices
        config.num_eval_envs = config.num_eval_envs // num_devices
        config.minibatch_size = config.minibatch_size // num_devices

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        max_episode_steps = config.env.max_episode_steps

        env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=max_episode_steps,
            parallel=config.num_envs,
            autoreset=True,
            fast_reset=True
        )

        # Maybe need a discount array for different agents
        agent = IMPALAAgent(
            action_space=env.action_space,
            obs_space=env.obs_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            normalize_obs=config.normalize_obs,
            continuous_action=config.agent_network.continuous_action,
            discount=config.discount,
            vtrace_lambda=config.vtrace_lambda,
            clip_rho_threshold=config.clip_rho_threshold,
            clip_c_threshold=config.clip_c_threshold,
            clip_pg_rho_threshold=config.clip_pg_rho_threshold,
            clip_ppo_epsilon=config.clip_ppo_epsilon,
            adv_mode=config.adv_mode,
            pg_loss_mode=config.pg_loss_mode,
        )

        if (config.optimizer.grad_clip_norm is not None and
                config.optimizer.grad_clip_norm > 0):
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.optimizer.grad_clip_norm),
                optax.adam(config.optimizer.lr)
            )
        else:
            optimizer = optax.adam(config.optimizer.lr)

        eval_env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset=False
        )

        evaluator = Evaluator(
            env=eval_env, agent=agent, max_episode_steps=max_episode_steps)

        return cls(env, agent, optimizer, evaluator, config)

    def step(self, state: State) -> tuple[IMPALATrainMetric, State]:

        key, rollout_key, learn_key, shuffle_key = jax.random.split(
            state.key, num=4)

        env_state, trajectory = rollout(
            self.env.step,
            self.agent.compute_actions,
            state.env_state,
            state.agent_state,
            rollout_key,
            rollout_length=self.config.rollout_length,
            env_extra_fields=('last_obs', 'episode_return')
        )

        agent_state = state.agent_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state, trajectory.obs,
                    pmap_axis_name=self.pmap_axis_name
                )
            )

        train_episode_return = average_episode_discount_return(
            trajectory.extras.env_extras.episode_return,
            trajectory.dones,
            pmap_axis_name=self.pmap_axis_name
        )

        trajectory = tree_stop_gradient(trajectory)

        def loss_fn(agent_state, sample_batch, key):
            # learn all data from trajectory
            loss_dict = self.agent.loss(agent_state, sample_batch, key)
            loss_weights = self.config.optimizer.loss_weights
            loss = jnp.zeros(())
            for loss_key in loss_weights.keys():
                loss += loss_weights[loss_key] * loss_dict[loss_key]

            return loss, loss_dict

        update_fn = agent_gradient_update(
            loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True
        )

        # minibatch_size: num of envs in one batch
        # unit in batch: trajectory [T, ...]
        num_minibatches = self.config.num_envs // self.config.minibatch_size

        def _get_shuffled_minibatch(perm_key, x):
            # x: [T, B, ...] -> [k, T, B//k, ...]
            x = jax.random.permutation(perm_key, x, axis=1)[
                :, :num_minibatches*self.config.minibatch_size]
            xs = jnp.stack(jnp.split(x, num_minibatches, axis=1))

            return xs

        def minibatch_step(carry, trajectory):
            opt_state, agent_state, key = carry
            key, learn_key = jax.random.split(key)

            (loss, loss_dict), agent_state, opt_state = update_fn(
                opt_state,
                agent_state,
                trajectory,
                learn_key
            )

            return (opt_state, agent_state, key), (loss, loss_dict)

        perm_key, learn_key = jax.random.split(key, num=2)

        (opt_state, agent_state, key), (loss_list, loss_dict_list) = jax.lax.scan(
            minibatch_step,
            (state.opt_state, agent_state, learn_key),
            jtu.tree_map(
                partial(_get_shuffled_minibatch, perm_key), trajectory),
            length=num_minibatches
        )

        # ======== update metrics ========

        loss = loss_list.mean()
        loss_dict = jtu.tree_map(jnp.mean, loss_dict_list)

        sampled_timesteps = psum(
            jnp.array(self.config.rollout_length *
                      self.config.num_envs, dtype=jnp.uint32),
            axis_name=self.pmap_axis_name)

        workflow_metrics = WorkflowMetric(
            sampled_timesteps=state.metrics.sampled_timesteps+sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        train_metrics = TrainMetric(
            train_episode_return=train_episode_return,
            loss=loss,
            raw_loss_dict=loss_dict
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state
        )

    def learn(self, state: State) -> State:
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        num_iters = math.ceil(self.config.total_timesteps / one_step_timesteps)

        start_iteration = tree_unpmap(
            state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            train_metrics = tree_unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = tree_unpmap(
                workflow_metrics, self.pmap_axis_name)

            self.recorder.write(workflow_metrics.to_local_dict(), i)
            train_metric_data = train_metrics.to_local_dict()
            if train_metrics.train_episode_return == MISSING_REWARD:
                train_metric_data['train_episode_return'] = None
            self.recorder.write(train_metric_data, i)

            if (i+1) % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)
                eval_metrics = tree_unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write({'eval': eval_metrics.to_local_dict()}, i)
                logger.debug(eval_metrics)

            self.checkpoint_manager.save(
                i,
                args=ocp.args.StandardSave(
                    tree_unpmap(state, self.pmap_axis_name))
            )

        return state


def compute_vtrace(
    rho_t, v_t, v_t_plus_1, rewards, dones,
    discount,
    lambda_=1.0,
    clip_rho_threshold=1.0,
    clip_c_threshold=1.0,
):
    chex.assert_trees_all_equal_shapes_and_dtypes(
        rho_t, v_t, v_t_plus_1, rewards, dones)

    discounts = discount * (1 - dones)

    # clip c and rho
    clipped_c_t = jnp.minimum(clip_c_threshold, rho_t) * lambda_
    clipped_rho_t = jnp.minimum(clip_rho_threshold, rho_t)

    # calculate δV_t
    td_error = clipped_rho_t * \
        (rewards + discounts * v_t_plus_1 - v_t)

    # calculate vtrace - v_t
    def _cal_vtrace_minus_v(vtrace_minus_v, params):
        td_error, discount, c = params
        vtrace_minus_v = td_error + discount * c * vtrace_minus_v
        return vtrace_minus_v, vtrace_minus_v

    _, vtrace_minus_v = jax.lax.scan(
        _cal_vtrace_minus_v,
        jnp.zeros_like(v_t[0]),
        (td_error, discounts, clipped_c_t),
        reverse=True
    )

    # calculate vs
    vtrace = vtrace_minus_v + v_t

    return vtrace


def compute_pg_advantage(
    rho_t, vtrace, v_t, v_t_plus_1, rewards, dones,
    discount,
    lambda_=1.0,
    clip_pg_rho_threshold=1.0,
    mode='official'
):
    discounts = discount * (1 - dones)
    # calculate advantage function
    if mode == 'official':
        # Note: rllib also follows this implementation
        q_t_plus_1 = jnp.concatenate([
            vtrace[1:],
            v_t_plus_1[-1:]
        ], axis=0)
        q_t = rewards + discounts*q_t_plus_1
    elif mode == 'acme':
        q_t_plus_1 = jnp.concatenate([
            lambda_ * vtrace[1:] + (1 - lambda_) * v_t[1:],
            v_t_plus_1[-1:]
        ], axis=0)
        q_t = rewards + discounts * q_t_plus_1
    else:
        raise ValueError(f'mode {mode} is not supported')

    clipped_pg_rho_t = jnp.minimum(clip_pg_rho_threshold, rho_t)
    pg_advantage = clipped_pg_rho_t * (q_t - v_t)

    return pg_advantage
