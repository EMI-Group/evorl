import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import struct
import math

from omegaconf import DictConfig


from evorl.sample_batch import SampleBatch
from evorl.networks import make_policy_network, make_value_network
from evorl.utils import running_statistics
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.toolkits import (
    compute_gae, flatten_rollout_trajectory,
    average_episode_discount_return, shuffle_sample_batch
)
from evorl.workflows import OffPolicyRLWorkflow
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
    MISSING_REWARD
)
from evorl.metrics import TrainMetric, WorkflowMetric
from typing import Tuple, Sequence, Optional, Any
import dataclasses
import wandb
import logging
import flax.linen as nn

import flashbax

logger = logging.getLogger(__name__)


@struct.dataclass
class IMPALANetworkParams:
    """Contains training state for the learner."""
    policy_params: Params
    value_params: Params

class IMPALATrainMetric(TrainMetric):
    rho: chex.Array = jnp.zeros((), dtype=jnp.float32)

class impalaAgent(Agent):
    actor_hidden_layer_sizes: Tuple[int] = (256, 256)
    critic_hidden_layer_sizes: Tuple[int] = (256, 256)
    normalize_obs: bool = False
    continuous_action: bool = False
    ppo_clipping_epsilon: float = 0.2
    clip_rho_threshold: float = 1.0
    clip_c_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
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

        policy_extras = PyTreeDict(
            # Log probabilities of the selected actions for importance sampling
            logp = actions_dist.log_prob(actions)
            # raw_action=raw_actions,
        )

        return jax.lax.stop_gradient(actions), policy_extras
    
    def get_action_log_prob(self, agent_state: AgentState, obs, action):
        # obs = sample_batch.obs
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
        action_log_prob = actions_dist.log_prob(action)
        
        return action_log_prob

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

        return jax.lax.stop_gradient(actions), PyTreeDict()

    def compute_values(self, agent_state: AgentState, sample_batch: SampleBatch) -> chex.Array:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        return self.value_network.apply(
            agent_state.params.value_params, obs).squeeze(-1)
    
    def compute_values_from_obs(self, agent_state: AgentState, obs) -> chex.Array:
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        return self.value_network.apply(
            agent_state.params.value_params, obs).squeeze(-1)
    
    def vtrace(self, v_t, v_t_plus_1, r_t, dones, discount_t, rho_t, lambda_ = 1.0, clip_rho_threshold = 1.0,
               clip_c_threshold = 1.0, clip_pg_rho_threshold = 1.0, stop_target_gradients = True):
        # chex.assert_rank([v_t, v_t_plus_1, r_t, discount_t, rho_t, lambda_],
        #                  [1, 1, 1, 1, 1, {0, 1}])
        chex.assert_type([v_t, v_t_plus_1, r_t, discount_t, rho_t, lambda_],
                        [float, float, float, float, float, float])
        chex.assert_equal_shape([v_t, v_t_plus_1, r_t, discount_t, rho_t])

        # update ratio of vs
        lambda_ = jnp.ones_like(discount_t) * lambda_

        # clip c and rho
        clipped_c_t = jnp.minimum(clip_c_threshold, rho_t) * lambda_
        clipped_rho_t = jnp.minimum(clip_rho_threshold, rho_t)

        # calculate δtV
        td_error = clipped_rho_t * (r_t + discount_t * (1 - dones) * v_t_plus_1 - v_t)

        # calculate vs - Vt
        def _cal_vs_minus_V(vs_minus_V, params):
            td_error, discount, c = params
            vs_minus_V = td_error + discount * c * vs_minus_V
            return vs_minus_V, vs_minus_V
        
        ini_vs_minus_V = jnp.zeros_like(discount_t[0])
        _, vs_minus_V = jax.lax.scan(_cal_vs_minus_V, ini_vs_minus_V, (td_error, discount_t*(1-dones), clipped_c_t), reverse = True)
        # vs_minus_V = vs_minus_V[:,0,:]

        # calculate vs
        vs = vs_minus_V + v_t

        #calculate advantage function
        q_bootstrap = jnp.concatenate([
            lambda_[:-1] * vs[1:] + (1 - lambda_[:-1]) * v_t[1:],
            v_t_plus_1[-1:]
        ], axis=0)
        q_value = r_t + discount_t * q_bootstrap
        clipped_pg_rho_t = jnp.minimum(clip_pg_rho_threshold, rho_t)
        pg_advantage = clipped_pg_rho_t * (q_value - v_t)
        
        # if stop_target_gradients:
        #     vs = jax.lax.stop_gradient(vs)
        
        return PyTreeDict(
            vs = jax.lax.stop_gradient(vs),
            pg_advantage = jax.lax.stop_gradient(pg_advantage),
        )
    def cal_temp_vtrace(self, agent_state: AgentState, trajectory: SampleBatch, discount,
                        clip_rho_threshold=1.0, clip_c_threshold=1.0, clip_pg_rho_threshold=1.0):
        # get obs from trajectory
        ob_t = trajectory.obs
        # ob_t_plus_1 = trajectory.next_obs
        last_obs = trajectory.extras.env_extras.last_obs
        v_obs = jnp.concatenate([trajectory.obs, last_obs[-1:]], axis=0)

        # calculate v with critic
        # v_t = self.compute_values_from_obs(agent_state, ob_t)
        # v_t_plus_1 = self.compute_values_from_obs(agent_state, ob_t_plus_1)
        vs = self.compute_values_from_obs(agent_state, v_obs)
        v_t = vs[:-1]
        v_t_plus_1 = vs[1:]

        # calculate importance ratio
        sampled_action_logits = trajectory.extras.policy_extras.logp
        action_t = trajectory.actions
        cal_action_logits = self.get_action_log_prob(agent_state, ob_t, action_t)
        rho_t = jnp.exp(cal_action_logits - sampled_action_logits)
        average_rho = jnp.mean(rho_t)
        dones = trajectory.dones

        # calculate vtrace
        r_t = trajectory.rewards
        discount_t = jnp.ones_like(r_t) * discount
        v_trace_dict = self.vtrace(v_t, v_t_plus_1, r_t, dones, discount_t, rho_t,
                                   clip_rho_threshold=clip_rho_threshold,
                                   clip_c_threshold=clip_c_threshold,
                                   clip_pg_rho_threshold=clip_pg_rho_threshold)

        return v_trace_dict, average_rho
    
    def sample_minibatch_from_trajectory(self, start_idx, end_idx, flatted_trajectory: SampleBatch):
        ob_t = flatted_trajectory.obs[start_idx: end_idx]
        action_t = flatted_trajectory.actions[start_idx: end_idx]
        extras_v = flatted_trajectory.extras.v_targets[start_idx: end_idx]
        extras_pg = flatted_trajectory.extras.advantages[start_idx: end_idx]
        logp = flatted_trajectory.extras.policy_extras.logp[start_idx: end_idx]
        policy_extras = PyTreeDict(logp = logp)
        extras = PyTreeDict(policy_extras = policy_extras, v_targets = extras_v, advantages = extras_pg)
        mini_sample_batch = SampleBatch(obs=ob_t, actions=action_t, extras=extras)

        return mini_sample_batch
    
    def loss_ppo(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
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

        v_targets = sample_batch.extras.v_targets

        value_loss = optax.l2_loss(vs, v_targets).mean()

        # ====== actor =======
        # PPO LOSS

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
        behavior_actions_logp = sample_batch.extras.policy_extras.logp

        advantages = sample_batch.extras.advantages

        rho = jnp.exp(actions_logp - behavior_actions_logp)

        # advantages: [T*B]
        policy_sorrogate_loss1 = rho * advantages
        policy_sorrogate_loss2 = jnp.clip(
            rho, 1-self.ppo_clipping_epsilon, 1+self.ppo_clipping_epsilon) * advantages
        policy_loss = - jnp.minimum(
            policy_sorrogate_loss1, policy_sorrogate_loss2).mean()
        # entropy: [T*B]
        if self.continuous_action:
            entropy_loss = actions_dist.entropy(seed=key).mean()
        else:
            entropy_loss = actions_dist.entropy().mean()

        return PyTreeDict(
            actor_loss=policy_loss,
            critic_loss=value_loss,
            actor_entropy_loss=entropy_loss
        )
    
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

        v_targets = sample_batch.extras.v_targets

        value_loss = optax.l2_loss(vs, v_targets).mean()

        # ====== actor =======
        # a2c LOSS

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

        advantages = sample_batch.extras.advantages

        # advantages: [T*B]
        policy_loss = - (advantages * actions_logp).mean()
        # entropy: [T*B]
        if self.continuous_action:
            entropy_loss = actions_dist.entropy(seed=key).mean()
        else:
            entropy_loss = actions_dist.entropy().mean()

        return PyTreeDict(
            actor_loss=policy_loss,
            critic_loss=value_loss,
            actor_entropy_loss=entropy_loss
        )

class IMPALAWorkflow(OnPolicyRLWorkflow):
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
        one_step_rollout_steps = config.num_envs * config.rollout_length
        if one_step_rollout_steps % config.minibatch_size != 0:
            logger.warning(
                f"minibatch_size ({config.minibath_size} cannot divides num_envs*rollout_length)")

        env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=max_episode_steps,
            parallel=config.num_envs,
            autoreset=True,
            fast_reset=True
        )

        # Maybe need a discount array for different agents
        agent = impalaAgent(
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
            config.env.env_name,
            config.env.env_type,
            episode_length=max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset=False
        )

        evaluator = Evaluator(
            env=eval_env, agent=agent, max_episode_steps=max_episode_steps)
        
        return cls(env, agent, optimizer, evaluator, config)
    
    def step(self, state: State) -> Tuple[IMPALATrainMetric, State]:

        key, rollout_key, learn_key, shuffle_key = jax.random.split(state.key, num=4)

        env_state, trajectory = rollout(
            self.env,
            self.agent,
            state.env_state,
            state.agent_state,
            rollout_key,
            rollout_length = self.config.rollout_length,
            discount=self.config.discount,
            env_extra_fields=('last_obs','episode_return')
        )

        agent_state = state.agent_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state, trajectory.obs,
                    pmap_axis_name=self.pmap_axis_name
                )
            )
        
        #-----------------compute vtrace for temp trajectory------------------------------
        v_trace_dict, rho_t = self.agent.cal_temp_vtrace(agent_state, trajectory, self.config.discount,
                                                  self.agent.clip_rho_threshold,
                                                  self.agent.clip_c_threshold,
                                                  self.agent.clip_pg_rho_threshold)
        trajectory.extras.v_targets = v_trace_dict.vs
        trajectory.extras.advantages = v_trace_dict.pg_advantage
        # [T,B,...]->[T*B,...]
        flatted_trajectory = flatten_rollout_trajectory(trajectory)
        flatted_trajectory = tree_stop_gradient(flatted_trajectory)

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

        num_minibatches = self.config.rollout_length * \
            self.config.num_envs // self.config.minibatch_size

        def _get_shuffled_minibatch(x):
            x = jax.random.permutation(shuffle_key, x)[
                :num_minibatches*self.config.minibatch_size]
            return x.reshape(num_minibatches, -1, *x.shape[1:])
        
        def _get_minibatch(x):
            return x.reshape(num_minibatches, -1, *x.shape[1:])

        temp_loss = 0
        temp_actor_loss = 0
        temp_critic_loss = 0
        temp_actor_entropy_loss = 0
        temp_rho = rho_t
        minibatch_size = self.config.minibatch_size

        opt_state = state.opt_state
        agent_state = agent_state
        key = learn_key

        for i in range(num_minibatches):
            # shuffled_minibatch = _get_shuffled_minibatch(flatten_rollout_trajectory)
            # shuffled_sample_batch = shuffle_sample_batch(trajectory, shuffle_key)
            flatted_shuffled_sample_batch = flatten_rollout_trajectory(trajectory)
            start_idx = i * minibatch_size
            end_idx = (i + 1) * minibatch_size
            mini_sample_batch = self.agent.sample_minibatch_from_trajectory(start_idx, end_idx,flatted_shuffled_sample_batch)
            mini_sample_batch = tree_stop_gradient(mini_sample_batch)

            (loss, loss_dict), opt_state, agent_state = update_fn(
                opt_state,
                agent_state,
                mini_sample_batch,
                learn_key
            )
            v_trace_dict, rho_t = self.agent.cal_temp_vtrace(agent_state, trajectory, self.config.discount,
                                        self.agent.clip_rho_threshold,
                                        self.agent.clip_c_threshold,
                                        self.agent.clip_pg_rho_threshold)
            trajectory.extras.v_targets = v_trace_dict.vs
            trajectory.extras.advantages = v_trace_dict.pg_advantage
            # [T,B,...]->[T*B,...]

            temp_loss +=  loss
            temp_actor_loss += loss_dict.actor_loss
            temp_critic_loss += loss_dict.critic_loss
            temp_actor_entropy_loss += loss_dict.actor_entropy_loss
            temp_rho += rho_t

        loss = temp_loss / num_minibatches
        rho = temp_rho / num_minibatches
        temp_actor_loss = temp_actor_loss / num_minibatches
        temp_critic_loss = temp_critic_loss / num_minibatches
        temp_actor_entropy_loss = temp_actor_entropy_loss / num_minibatches
        loss_dict = PyTreeDict(
            actor_loss=temp_actor_loss,
            critic_loss=temp_critic_loss,
            actor_entropy_loss=temp_actor_entropy_loss
        )

        # ======== update metrics ========

        sampled_timesteps = psum(self.config.rollout_length * self.config.num_envs,
                                 axis_name=self.pmap_axis_name)

        train_episode_return = average_episode_discount_return(
            trajectory.extras.env_extras.episode_return,
            trajectory.dones,
            pmap_axis_name=self.pmap_axis_name
        )

        workflow_metrics = WorkflowMetric(
            sampled_timesteps=state.metrics.sampled_timesteps+sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        train_metrics = IMPALATrainMetric(
            train_episode_return=train_episode_return,
            loss=loss,
            raw_loss_dict=loss_dict,
            rho=rho
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.update(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state
        )
        #=========================================================================

    def learn(self, state: State) -> State:
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        num_iters = math.ceil(self.config.total_timesteps / one_step_timesteps)

        start_iteration = tree_unpmap(state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            train_metrics = tree_unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = tree_unpmap(workflow_metrics, self.pmap_axis_name)

            self.recorder.write(workflow_metrics.to_local_dict(), i)
            train_metric_data = train_metrics.to_local_dict()
            if train_metrics.train_episode_return == MISSING_REWARD:
                del train_metric_data['train_episode_return']
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
        extras=PyTreeDict(
            policy_extras=policy_extras,
            env_extras=env_extras
        ))

    return env_nstate, transition


def rollout(
    env: Env,
    agent: impalaAgent,
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
        truncation = env_nstate.info.truncation  # [#envs]
        # Note: if truncation happens in any env in the batch, apply PEB
        rewards = transition.rewards + discount * jax.lax.cond(
            truncation.any(),
            lambda last_obs: agent.compute_values(
                agent_state, SampleBatch(obs=last_obs)) * truncation,
            lambda last_obs: jnp.zeros_like(transition.rewards),
            env_nstate.info.last_obs  # [#envs, ...]
        )

        transition = transition.replace(rewards=rewards)
        # transition.info["policy_extras"]["peb_rewards"] = reward # ok for dict

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key), (), length=rollout_length)

    return env_state, trajectory