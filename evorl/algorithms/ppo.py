import logging
import math
from functools import partial
from typing import Any
from omegaconf import DictConfig

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.distributed import agent_gradient_update, psum, unpmap
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.envs import AutoresetMode, create_env, Space, Box, Discrete
from evorl.evaluators import Evaluator
from evorl.metrics import TrainMetric, MetricBase
from evorl.networks import make_policy_network, make_v_network
from evorl.rollout import rollout
from evorl.sample_batch import SampleBatch
from evorl.types import (
    MISSING_REWARD,
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
from evorl.utils.jax_utils import (
    tree_get,
    tree_stop_gradient,
    scan_and_mean,
)
from evorl.utils.rl_toolkits import (
    average_episode_discount_return,
    compute_gae_with_horizon,
    flatten_rollout_trajectory,
    approximate_kl,
)
from evorl.workflows import OnPolicyWorkflow
from evorl.recorders import add_prefix
from evorl.agent import Agent, AgentState

logger = logging.getLogger(__name__)


class PPONetworkParams(PyTreeData):
    """Contains training state for the learner."""

    policy_params: Params
    value_params: Params


class PPOAgent(Agent):
    continuous_action: bool
    policy_network: nn.Module
    value_network: nn.Module
    obs_preprocessor: Any = pytree_field(default=None, static=True)

    clip_epsilon: float = 0.2
    normalize_gae: bool = True
    policy_obs_key: str = ""
    value_obs_key: str = ""

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        policy_key, value_key = jax.random.split(key, 2)

        dummy_obs = jtu.tree_map(lambda x: x[None, ...], obs_space.sample(key))

        policy_params = self.policy_network.init(policy_key, dummy_obs)

        value_params = self.value_network.init(value_key, dummy_obs)

        params_state = PPONetworkParams(
            policy_params=policy_params, value_params=value_params
        )

        if self.normalize_obs:
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(
                tree_get(dummy_obs, 0)
            )
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state, obs_preprocessor_state=obs_preprocessor_state
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.sample(seed=key)

        policy_extras = PyTreeDict(
            # raw_action=raw_actions,
            logp=actions_dist.log_prob(actions)
        )

        return actions, policy_extras

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.mode()

        return actions, PyTreeDict()

    def loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # mask invalid transitions at autoreset
        mask = jnp.logical_not(sample_batch.extras.env_extras.autoreset)

        # ======= critic =======
        vs = self.value_network.apply(agent_state.params.value_params, obs)

        v_targets = sample_batch.extras.v_targets

        critic_loss = optax.squared_error(vs, v_targets).mean(where=mask)

        # ====== actor =======

        # [T*B, A]
        raw_actions = self.policy_network.apply(agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        # [T*B]
        actions_logp = actions_dist.log_prob(sample_batch.actions)
        behavior_actions_logp = sample_batch.extras.policy_extras.logp

        advantages = sample_batch.extras.advantages
        if self.normalize_gae:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logrho = actions_logp - behavior_actions_logp
        rho = jnp.exp(logrho)

        # advantages: [T*B]
        policy_sorrogate_loss1 = rho * advantages
        policy_sorrogate_loss2 = (
            jnp.clip(rho, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        )
        actor_loss = -jnp.minimum(policy_sorrogate_loss1, policy_sorrogate_loss2).mean(
            where=mask
        )

        # entropy: [T*B]
        if self.continuous_action:
            actor_entropy = actions_dist.entropy(seed=key).mean(where=mask)
        else:
            actor_entropy = actions_dist.entropy().mean(where=mask)

        approx_kl = approximate_kl(logrho)

        return PyTreeDict(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            actor_entropy=actor_entropy,
            approx_kl=approx_kl,
        )

    def compute_values(
        self, agent_state: AgentState, sample_batch: SampleBatch
    ) -> chex.Array:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        return self.value_network.apply(agent_state.params.value_params, obs)


def make_mlp_ppo_agent(
    action_space: Space,
    clip_epsilon: float = 0.2,
    actor_hidden_layer_sizes: tuple[int] = (256, 256),
    critic_hidden_layer_sizes: tuple[int] = (256, 256),
    normalize_obs: bool = False,
    normalize_gae: bool = False,
    policy_obs_key: str = "",
    value_obs_key: str = "",
):
    if isinstance(action_space, Box):
        action_size = action_space.shape[0] * 2
        continuous_action = True
    elif isinstance(action_space, Discrete):
        action_size = action_space.n
        continuous_action = False
    else:
        raise NotImplementedError(f"Unsupported action space: {action_space}")

    policy_network = make_policy_network(
        action_size=action_size,
        hidden_layer_sizes=actor_hidden_layer_sizes,
        obs_key=policy_obs_key,
    )

    value_network = make_v_network(
        hidden_layer_sizes=critic_hidden_layer_sizes,
        obs_key=value_obs_key,
    )

    if normalize_obs:
        obs_preprocessor = running_statistics.normalize
    else:
        obs_preprocessor = None

    return PPOAgent(
        continuous_action=continuous_action,
        policy_network=policy_network,
        value_network=value_network,
        obs_preprocessor=obs_preprocessor,
        clip_epsilon=clip_epsilon,
        normalize_gae=normalize_gae,
        policy_obs_key=policy_obs_key,
        value_obs_key=value_obs_key,
    )


class PPOWorkflow(OnPolicyWorkflow):
    @classmethod
    def name(cls):
        return "PPO"

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
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
                f"rescale minibatch_size to {config.minibatch_size // num_devices}"
            )

        config.num_envs = config.num_envs // num_devices
        config.num_eval_envs = config.num_eval_envs // num_devices
        config.minibatch_size = config.minibatch_size // num_devices

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        max_episode_steps = config.env.max_episode_steps

        env = create_env(
            config.env,
            episode_length=max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.ENVPOOL,
        )

        agent = make_mlp_ppo_agent(
            action_space=env.action_space,
            clip_epsilon=config.clip_epsilon,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            normalize_obs=config.normalize_obs,
            normalize_gae=config.normalize_gae,
            policy_obs_key=config.agent_network.policy_obs_key,
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

        eval_env = create_env(
            config.env,
            episode_length=max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        one_step_rollout_steps = config.num_envs * config.rollout_length
        if one_step_rollout_steps % config.minibatch_size != 0:
            logger.warning(
                f"minibatch_size ({config.minibath_size} cannot divides num_envs*rollout_length)"
            )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=max_episode_steps,
        )

        return cls(env, agent, optimizer, evaluator, config)

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        # trajectory: [T, #envs, ...]
        trajectory, env_state = rollout(
            self.env.step,
            self.agent.compute_actions,
            state.env_state,
            state.agent_state,
            rollout_key,
            rollout_length=self.config.rollout_length,
            env_extra_fields=("autoreset", "episode_return", "termination"),
        )

        agent_state = state.agent_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    pmap_axis_name=self.pmap_axis_name,
                )
            )

        train_episode_return = average_episode_discount_return(
            trajectory.extras.env_extras.episode_return,
            trajectory.dones,
            pmap_axis_name=self.pmap_axis_name,
        )

        # ======== compute GAE =======
        _obs = jtu.tree_map(
            lambda obs, next_obs: jnp.concatenate([obs, next_obs[-1:]], axis=0),
            trajectory.obs,
            trajectory.next_obs,
        )
        # concat [values, bootstrap_value]
        vs = self.agent.compute_values(state.agent_state, SampleBatch(obs=_obs))

        v_targets, advantages = compute_gae_with_horizon(
            rewards=trajectory.rewards,
            values=vs,
            dones=trajectory.dones,
            terminations=trajectory.extras.env_extras.termination,
            gae_horizon=self.config.gae_horizon,
            gae_lambda=self.config.gae_lambda,
            discount=self.config.discount,
        )

        trajectory.extras.v_targets = jax.lax.stop_gradient(v_targets)
        trajectory.extras.advantages = jax.lax.stop_gradient(advantages)
        # [T,B,...] -> [T*B,...]
        trajectory = tree_stop_gradient(flatten_rollout_trajectory(trajectory))
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
            loss_fn, self.optimizer, pmap_axis_name=self.pmap_axis_name, has_aux=True
        )

        num_minibatches = (
            self.config.rollout_length
            * self.config.num_envs
            // self.config.minibatch_size
        )

        def _get_shuffled_minibatch(perm_key, x):
            x = x[jax.random.permutation(perm_key, x.shape[0])][
                : num_minibatches * self.config.minibatch_size
            ]
            return x.reshape(num_minibatches, self.config.minibatch_size, *x.shape[1:])

        def minibatch_step(carry, trajectory):
            opt_state, agent_state, key = carry
            key, learn_key = jax.random.split(key)

            (loss, loss_dict), agent_state, opt_state = update_fn(
                opt_state, agent_state, trajectory, learn_key
            )

            return (opt_state, agent_state, key), (loss, loss_dict)

        def epoch_step(carry, _):
            opt_state, agent_state, key = carry
            perm_key, learn_key = jax.random.split(key, num=2)

            (opt_state, agent_state, key), (loss, loss_dict) = scan_and_mean(
                minibatch_step,
                (opt_state, agent_state, learn_key),
                jtu.tree_map(partial(_get_shuffled_minibatch, perm_key), trajectory),
                length=num_minibatches,
            )

            return (opt_state, agent_state, key), (loss, loss_dict)

        # loss_list: [reuse_rollout_epochs, num_minibatches]
        (opt_state, agent_state, _), (loss, loss_dict) = scan_and_mean(
            epoch_step,
            (state.opt_state, agent_state, learn_key),
            None,
            length=self.config.reuse_rollout_epochs,
        )

        # ======== update metrics ========

        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.pmap_axis_name,
        )

        sampled_epsiodes = psum(
            trajectory.dones.sum().astype(jnp.uint32), axis_name=self.pmap_axis_name
        )

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_epsiodes,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        train_metrics = TrainMetric(
            train_episode_return=train_episode_return,
            loss=loss,
            raw_loss_dict=loss_dict,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state,
        )

    def learn(self, state: State) -> State:
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        num_iters = math.ceil(self.config.total_timesteps / one_step_timesteps)

        start_iteration = unpmap(state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            iters = i + 1
            train_metrics = unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = unpmap(workflow_metrics, self.pmap_axis_name)

            self.recorder.write(workflow_metrics.to_local_dict(), iters)
            train_metric_data = train_metrics.to_local_dict()
            if train_metrics.train_episode_return == MISSING_REWARD:
                train_metric_data["train_episode_return"] = None
            self.recorder.write(train_metric_data, iters)

            if iters % self.config.eval_interval == 0 or iters == num_iters:
                eval_metrics, state = self.evaluate(state)
                eval_metrics = unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iters
                )

            self.checkpoint_manager.save(
                iters,
                unpmap(state, self.pmap_axis_name),
                force=iters == num_iters,
            )

        return state
