import jax
import jax.numpy as jnp
from flax import struct
import math

from omegaconf import DictConfig

from evorl.types import SampleBatch, metricfield

from evorl.networks import make_policy_network, make_value_network
from evorl.utils import running_statistics
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.toolkits import (
    compute_gae, flatten_rollout_trajectory,
    average_episode_discount_return
)
from evorl.workflows import OnPolicyRLWorkflow
from evorl.agents import AgentState
from evorl.distributed import agent_gradient_update, pmean
from evorl.envs import create_env, Env
from evorl.evaluator import Evaluator
from .agent import Agent, AgentState

from evox import State

# from typing import Dict
import chex
import optax
from evorl.types import (
    LossDict, Action, Params, PolicyExtraInfo, EnvState,
    TrainMetric, EvaluateMetric, WorkflowMetric
)
from typing import Tuple, Sequence, Optional
import dataclasses

import logging

logger = logging.getLogger(__name__)


@struct.dataclass
class A2CNetworkParams:
    """Contains training state for the learner."""
    policy_params: Params
    value_params: Params


@dataclasses.dataclass(unsafe_hash=True)
class A2CAgent(Agent):
    actor_hidden_layer_sizes: Tuple[int] = (256, 256)
    critic_hidden_layer_sizes: Tuple[int] = (256, 256)
    normalize_obs: bool = False
    continuous_action: bool = False

    def init(self, key: chex.PRNGKey) -> AgentState:

        obs_size = self.obs_space.shape[0]

        if self.continuous_action:
            action_size = self.action_space.shape[0]
            action_size *= 2
        else:
            action_size = self.action_space.n

        policy_key, value_key, obs_preprocessor_key = jax.random.split(key, 3)
        self.policy_network, policy_init_fn = make_policy_network(
            action_size=action_size,
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes
        )
        policy_params = policy_init_fn(policy_key)

        self.value_network, value_init_fn = make_value_network(
            obs_size=obs_size,
            hidden_layer_sizes=self.critic_hidden_layer_sizes
        )
        value_params = value_init_fn(value_key)

        params_state = A2CNetworkParams(
            policy_params=policy_params,
            value_params=value_params
        )

        if self.normalize_obs:
            self.obs_preprocessor = running_statistics.normalize
            dummy_obs = self.obs_space.sample(obs_preprocessor_key)
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state
        )

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
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

        policy_extras = dict(
            raw_action=raw_actions,
            logp=actions_dist.log_prob(actions)
        )

        return jax.lax.stop_gradient(actions), policy_extras

    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
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

        return jax.lax.stop_gradient(actions), {}

    def loss(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
        """

            sample_batch: [T*B, ...]


            Return: LossDict[
                actor_loss
                critic_loss
                actor_entropy_loss
            ]
        """

        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        # ======= critic =======
        vs = self.value_network.apply(
            agent_state.params.value_params, obs).squeeze(-1)

        v_targets = sample_batch.extras['v_targets']

        value_loss = optax.huber_loss(vs, v_targets, delta=1).mean()

        # ====== actor =======

        # [T*B, A]
        raw_actions = self.policy_network.apply(
            agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(
                *jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        # [T*B]
        actions_logp = actions_dist.log_prob(sample_batch.actions)

        advantages = sample_batch.extras['advantages']

        # advantages: [T*B]
        policy_loss = - (advantages * actions_logp).mean()
        # entropy: [T*B]
        if self.continuous_action:
            entropy_loss = actions_dist.entropy(seed=key).mean()
        else:
            entropy_loss = actions_dist.entropy().mean()

        return dict(
            actor_loss=policy_loss,
            critic_loss=value_loss,
            actor_entropy_loss=entropy_loss
        )

    def compute_values(self, agent_state: AgentState, sample_batch: SampleBatch) -> chex.Array:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        return self.value_network.apply(
            agent_state.params.value_params, obs).squeeze(-1)


class A2CWorkflow(OnPolicyRLWorkflow):
    @staticmethod
    def _rescale_config(config, devices) -> None:
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

        config.num_envs = config.num_envs // num_devices
        config.num_eval_envs = config.num_eval_envs // num_devices

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env,
            config.env_type,
            episode_length=1000,
            parallel=config.num_envs,
            autoreset=True
        )

        agent = A2CAgent(
            action_space=env.action_space,
            obs_space=env.obs_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            normalize_obs=config.normalize_obs,
            continuous_action=config.agent_network.continuous_action
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
            config.env,
            config.env_type,
            episode_length=1000,
            parallel=config.num_eval_envs,
            autoreset=False
        )

        evaluator = Evaluator(
            env=eval_env, agent=agent, max_episode_length=1000)

        return cls(env, agent, optimizer, evaluator, config)

    def step(self, state: State) -> Tuple[TrainMetric, State]:

        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        # trajectory: [T, #envs, ...]
        env_state, trajectory = rollout(
            self.env,
            self.agent,
            state.env_state,
            state.agent_state,
            rollout_key,
            rollout_length=self.config.rollout_length,
            discount=self.config.discount,
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

        # ======== compute GAE =======
        last_obs = trajectory.extras['env_extras']['last_obs']
        v_obs = jnp.concatenate(
            [trajectory.obs, last_obs[-1:]], axis=0
        )
        # concat [values, bootstrap_value]
        vs = self.agent.compute_values(
            state.agent_state, SampleBatch(obs=v_obs))
        v_targets, advantages = compute_gae(
            rewards=trajectory.rewards,  # peb_rewards
            values=vs,
            dones=trajectory.dones,
            gae_lambda=self.config.gae_lambda,
            discount=self.config.discount
        )

        trajectory.extras['v_targets'] = v_targets
        trajectory.extras['advantages'] = advantages
        # [T,B,...] -> [T*B,...]
        trajectory = flatten_rollout_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)
        # ============================

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
            has_aux=True)

        (loss, loss_dict), opt_state, agent_state = update_fn(
            state.opt_state,
            agent_state,
            trajectory,
            learn_key
        )

        # ======== update metrics ========

        sampled_timesteps = (state.metrics.sampled_timesteps +
                         self.config.rollout_length * self.config.num_envs)

        train_episode_return = average_episode_discount_return(
            trajectory.extras["env_extras"]['episode_return'],
            trajectory.dones
        ).mean()

        workflow_metrics = WorkflowMetric(
            sampled_timesteps=sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        train_metrics = TrainMetric(
            train_episode_return=train_episode_return,
            loss=loss,
            raw_loss_dict=loss_dict
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.update(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state
        )

    def learn(self, state: State) -> State:
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        num_iters = math.ceil(self.config.total_timesteps / one_step_timesteps)

        for i in range(num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            # logger.info(workflow_metrics)
            # logger.info(train_metrics)

            if (i+1) % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)
                logger.info(eval_metrics)

        return state


def env_step(
    env: Env,
    agent: Agent,
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    sample_batch: SampleBatch,
    key: chex.PRNGKey,
    env_extra_fields: Sequence[str] = (),
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect data.
    """

    actions, policy_extras = agent.compute_actions(
        agent_state, sample_batch, key)
    env_nstate = env.step(env_state, actions)

    info = env_nstate.info
    env_extras = {x: info[x] for x in env_extra_fields if x in info}

    transition = SampleBatch(
        obs=env_state.obs,
        actions=actions,
        rewards=env_nstate.reward,
        dones=env_nstate.done,
        # next_obs=env_nstate.info["last_obs"],
        next_obs=env_nstate.obs,
        extras=dict(
            policy_extras=policy_extras,
            env_extras=env_extras
        ))

    return env_nstate, transition


def rollout(
    env: Env,
    agent: A2CAgent,
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    discount: float,
    env_extra_fields: Sequence[str] = ('last_obs',),
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect given rollout_length trajectory.

        Args:
            env: vampped env w/ autoreset
        Returns:
            env_state: last env_state after rollout
            trajectory: SampleBatch [T, #envs, ...], T=rollout_length
    """

    def _one_step_rollout(carry, unused_t):
        """
            sample_batch: one-step obs
            transition: one-step full info
        """
        env_state, current_key = carry
        next_key, current_key = jax.random.split(current_key, 2)

        # sample_batch: [#envs, ...]
        sample_batch = SampleBatch(
            obs=env_state.obs,
        )

        # transition: [#envs, ...]
        env_nstate, transition = env_step(
            env, agent, env_state, agent_state,
            sample_batch, current_key, env_extra_fields
        )

        # set PEB reward for GAE:
        truncation = env_nstate.info['truncation']  # [#envs]
        # Note: if truncation happens in any env in the batch, apply PEB
        rewards = transition.rewards + discount * jax.lax.cond(
            truncation.any(),
            lambda last_obs: agent.compute_values(
                agent_state, SampleBatch(obs=last_obs)) * truncation,
            lambda last_obs: jnp.zeros_like(transition.rewards),
            env_nstate.info["last_obs"]  # [#envs, ...]
        )

        transition = transition.replace(rewards=rewards)
        # transition.info["policy_extras"]["peb_rewards"] = reward # ok for dict

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key), (), length=rollout_length)

    return env_state, trajectory
