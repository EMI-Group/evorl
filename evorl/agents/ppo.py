import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import orbax.checkpoint as ocp
import chex
import optax
import math
from omegaconf import DictConfig
import flax.linen as nn
from functools import partial


from evorl.sample_batch import SampleBatch
from evorl.networks import make_policy_network, make_v_network
from evorl.utils import running_statistics
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.utils.jax_utils import tree_stop_gradient, rng_split
from evorl.utils.toolkits import (
    compute_gae, flatten_rollout_trajectory,
    average_episode_discount_return
)
from evorl.workflows import OnPolicyRLWorkflow
from evorl.agents import AgentState
from evorl.distributed import agent_gradient_update, tree_unpmap, psum
from evorl.envs import create_env, Env, EnvState
from evorl.evaluator import Evaluator
from evorl.rollout import env_step
from evorl.types import (
    LossDict, Action, Params, PolicyExtraInfo, PyTreeDict, pytree_field,
    MISSING_REWARD, PyTreeData, State
)
from evorl.metrics import TrainMetric, WorkflowMetric
from .agent import Agent, AgentState

from collections.abc import Sequence
from typing import Any
import logging


logger = logging.getLogger(__name__)


class PPONetworkParams(PyTreeData):
    """Contains training state for the learner."""
    policy_params: Params
    value_params: Params


class PPOAgent(Agent):
    actor_hidden_layer_sizes: tuple[int] = (256, 256)
    critic_hidden_layer_sizes: tuple[int] = (256, 256)
    normalize_obs: bool = False
    continuous_action: bool = False
    clip_epsilon: float = 0.2
    policy_network: nn.Module = pytree_field(lazy_init=True)  # nn.Module is ok
    value_network: nn.Module = pytree_field(lazy_init=True)
    obs_preprocessor: Any = pytree_field(lazy_init=True, pytree_node=False)

    def init(self, key: chex.PRNGKey) -> AgentState:

        obs_size = self.obs_space.shape[0]

        if self.continuous_action:
            action_size = self.action_space.shape[0] * 2
        else:
            action_size = self.action_space.n

        policy_key, value_key, obs_preprocessor_key = jax.random.split(key, 3)
        policy_network, policy_init_fn = make_policy_network(
            action_size=action_size,
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes
        )
        policy_params = policy_init_fn(policy_key)

        value_network, value_init_fn = make_v_network(
            obs_size=obs_size,
            hidden_layer_sizes=self.critic_hidden_layer_sizes
        )
        value_params = value_init_fn(value_key)

        self.set_frozen_attr('policy_network', policy_network)
        self.set_frozen_attr('value_network', value_network)

        params_state = PPONetworkParams(
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
            # raw_action=raw_actions,
            logp=actions_dist.log_prob(actions)
        )

        return actions, policy_extras

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
            agent_state.params.value_params, obs)

        v_targets = sample_batch.extras.v_targets

        critic_loss = optax.l2_loss(vs, v_targets).mean()

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
        behavior_actions_logp = sample_batch.extras.policy_extras.logp

        advantages = sample_batch.extras.advantages

        rho = jnp.exp(actions_logp-behavior_actions_logp)

        # advantages: [T*B]
        policy_sorrogate_loss1 = rho * advantages
        policy_sorrogate_loss2 = jnp.clip(
            rho, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        policy_loss = - jnp.minimum(
            policy_sorrogate_loss1, policy_sorrogate_loss2).mean()

        # entropy: [T*B]
        if self.continuous_action:
            entropy_loss = actions_dist.entropy(seed=key).mean()
        else:
            entropy_loss = actions_dist.entropy().mean()

        return PyTreeDict(
            actor_loss=policy_loss,
            critic_loss=critic_loss,
            actor_entropy_loss=entropy_loss
        )

    def compute_values(self, agent_state: AgentState, sample_batch: SampleBatch) -> chex.Array:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        return self.value_network.apply(
            agent_state.params.value_params, obs)


class PPOWorkflow(OnPolicyRLWorkflow):
    @classmethod
    def name(cls):
        return "PPO"

    @staticmethod
    def _rescale_config(config: DictConfig) -> None:
        num_devices = jax.device_count()

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
            # fast_reset=True
        )

        agent = PPOAgent(
            action_space=env.action_space,
            obs_space=env.obs_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            normalize_obs=config.normalize_obs,
            continuous_action=config.agent_network.continuous_action,
            clip_epsilon=config.clip_epsilon
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

        one_step_rollout_steps = config.num_envs * config.rollout_length
        if one_step_rollout_steps % config.minibatch_size != 0:
            logger.warning(
                f"minibatch_size ({config.minibath_size} cannot divides num_envs*rollout_length)")

        evaluator = Evaluator(
            env=eval_env, agent=agent, max_episode_steps=max_episode_steps)

        return cls(env, agent, optimizer, evaluator, config)

    def step(self, state: State) -> tuple[TrainMetric, State]:

        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        # trajectory: [T, #envs, ...]
        trajectory, env_state = rollout(
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

        train_episode_return = average_episode_discount_return(
            trajectory.extras.env_extras.episode_return,
            trajectory.dones,
            pmap_axis_name=self.pmap_axis_name
        )

        # ======== compute GAE =======
        last_obs = trajectory.extras.env_extras.last_obs
        _obs = jnp.concatenate(
            [trajectory.obs, last_obs[-1:]], axis=0
        )
        # concat [values, bootstrap_value]
        vs = self.agent.compute_values(
            state.agent_state, SampleBatch(obs=_obs))
        v_targets, advantages = compute_gae(
            rewards=trajectory.rewards,  # peb_rewards
            values=vs,
            dones=trajectory.dones,
            gae_lambda=self.config.gae_lambda,
            discount=self.config.discount
        )
        trajectory.extras.v_targets = jax.lax.stop_gradient(v_targets)
        trajectory.extras.advantages = jax.lax.stop_gradient(advantages)
        # [T,B,...] -> [T*B,...]
        trajectory = tree_stop_gradient(
            flatten_rollout_trajectory(trajectory)
        )
        # ============================

        def loss_fn(agent_state, sample_batch, key):
            # learn all data from trajectory
            loss_dict = self.agent.loss(agent_state, sample_batch, key)
            loss_weights = self.config.loss_weights
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

        num_minibatches = self.config.rollout_length * \
            self.config.num_envs // self.config.minibatch_size

        def _get_shuffled_minibatch(perm_key, x):
            x = jax.random.permutation(perm_key, x)[
                :num_minibatches*self.config.minibatch_size]
            return x.reshape(num_minibatches, -1, *x.shape[1:])

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

        def epoch_step(carry, _):
            opt_state, agent_state, key = carry
            perm_key, learn_key = jax.random.split(key, num=2)

            (opt_state, agent_state, key), (loss_list, loss_dict_list) = jax.lax.scan(
                minibatch_step,
                (opt_state, agent_state, learn_key),
                jtu.tree_map(
                    partial(_get_shuffled_minibatch, perm_key), trajectory),
                length=num_minibatches
            )

            return (opt_state, agent_state, key), (loss_list, loss_dict_list)

        # loss_list: [reuse_rollout_epochs, num_minibatches]
        (opt_state, agent_state, _), (loss_list, loss_dict_list) = jax.lax.scan(
            epoch_step,
            (state.opt_state, agent_state, learn_key),
            None,
            length=self.config.reuse_rollout_epochs
        )

        loss = loss_list.mean()
        loss_dict = jtu.tree_map(jnp.mean, loss_dict_list)

        # ======== update metrics ========

        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.pmap_axis_name)

        workflow_metrics = state.metrics.replace(
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


def rollout(
    env: Env,
    agent: PPOAgent,
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    discount: float,
    env_extra_fields: Sequence[str] = ('last_obs',),
) -> tuple[SampleBatch, EnvState]:
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
        next_key, current_key = rng_split(current_key, 2)

        # sample_batch: [#envs, ...]
        sample_batch = SampleBatch(
            obs=env_state.obs,
        )

        # transition: [#envs, ...]
        transition, env_nstate = env_step(
            env.step, agent.compute_actions,
            env_state, agent_state,
            sample_batch, current_key, env_extra_fields
        )

        # set PEB reward for GAE:
        truncation = env_nstate.info.truncation  # [#envs]
        # Note: if truncation happens in any env in the batch, apply PEB for episodes with truncation=1
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
