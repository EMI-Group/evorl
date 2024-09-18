import logging
from collections.abc import Callable
from typing import Any

import chex
import flashbax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig

from evorl.distributed import agent_gradient_update, psum, tree_pmean
from evorl.distribution import get_tanh_norm_dist
from evorl.envs import AutoresetMode, Box, create_env, Space
from evorl.evaluator import Evaluator
from evorl.metrics import MetricBase, TrainMetric, metricfield
from evorl.networks import make_policy_network, make_q_network
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
from evorl.utils.jax_utils import (
    scan_and_mean,
    tree_stop_gradient,
)
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update

from evorl.agent import Agent, AgentState
from .offpolicy_utils import OffPolicyWorkflowTemplate, clean_trajectory

logger = logging.getLogger(__name__)
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


class SACTrainMetric(MetricBase):
    critic_loss: chex.Array
    actor_loss: chex.Array
    alpha_loss: chex.Array | None = None
    raw_loss_dict: LossDict = metricfield(
        default_factory=PyTreeDict, reduce_fn=tree_pmean
    )


class SACNetworkParams(PyTreeData):
    critic_params: Params
    target_critic_params: Params
    actor_params: Params
    log_alpha: Params


class SACAgent(Agent):
    critic_hidden_layer_sizes: tuple[int] = (256, 256)
    actor_hidden_layer_sizes: tuple[int] = (256, 256)
    init_alpha: float = 1.0
    discount: float = 0.99
    reward_scale: float = 1.0
    normalize_obs: bool = False
    critic_network: nn.Module = pytree_field(lazy_init=True)
    actor_network: nn.Module = pytree_field(lazy_init=True)
    obs_preprocessor: Any = pytree_field(lazy_init=True, pytree_node=False)

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        obs_size = obs_space.shape[0]
        action_size = action_space.shape[0]

        key, critic_key, actor_key, obs_preprocessor_key = jax.random.split(key, num=4)

        critic_network, critic_init_fn = make_q_network(
            obs_size=obs_size,
            action_size=action_size,
            n_stack=2,
            hidden_layer_sizes=self.critic_hidden_layer_sizes,
        )

        critic_params = critic_init_fn(critic_key)
        target_critic_params = critic_params

        actor_network, actor_init_fn = make_policy_network(
            action_size=action_size * 2,  # mean+std
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes,
        )

        actor_params = actor_init_fn(actor_key)

        log_alpha = jnp.log(jnp.float32(self.init_alpha))

        self.set_frozen_attr("critic_network", critic_network)
        self.set_frozen_attr("actor_network", actor_network)

        params_state = SACNetworkParams(
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            actor_params=actor_params,
            log_alpha=log_alpha,
        )

        if self.normalize_obs:
            obs_preprocessor = running_statistics.normalize
            self.set_frozen_attr("obs_preprocessor", obs_preprocessor)
            dummy_obs = obs_space.sample(obs_preprocessor_key)
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
        else:
            obs_preprocessor_state = None

        entropy_target = -jnp.prod(jnp.array(action_space.shape, dtype=jnp.float32))

        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state,
            extra_state=PyTreeDict(entropy_target=entropy_target),  # the constant
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.actor_network.apply(agent_state.params.actor_params, obs)
        actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        actions = actions_dist.sample(seed=key)
        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.actor_network.apply(agent_state.params.actor_params, obs)
        actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        actions = actions_dist.mode()
        return actions, PyTreeDict()

    def alpha_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.actor_network.apply(agent_state.params.actor_params, obs)
        actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        actions = actions_dist.sample(seed=key)
        actions_logp = actions_dist.log_prob(actions)

        entropy_target = agent_state.extra_state.entropy_target
        # official impl:
        alpha = jnp.exp(agent_state.params.log_alpha)
        alpha_loss = jnp.mean(
            -alpha * jax.lax.stop_gradient(actions_logp + entropy_target)
        )

        # another impl: see stable-baselines3/issues/36
        # alpha_loss = (- agent_state.params.log_alpha *
        #               jax.lax.stop_gradient(actions_logp + entropy_target)).mean()

        return PyTreeDict(
            alpha_loss=alpha_loss, log_alpha=agent_state.params.log_alpha, alpha=alpha
        )

    def actor_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        actor_key, entropy_key = jax.random.split(key, 2)
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        alpha = jnp.exp(agent_state.params.log_alpha)

        raw_actions = self.actor_network.apply(agent_state.params.actor_params, obs)
        actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        actions = actions_dist.sample(seed=actor_key)
        actions_logp = actions_dist.log_prob(actions)

        # [B, 2]
        q_values = self.critic_network.apply(
            agent_state.params.critic_params, obs, actions
        )
        min_q = jnp.min(q_values, axis=-1)
        actor_loss = jnp.mean(alpha * actions_logp - min_q)
        entropy = actions_dist.entropy(seed=entropy_key).mean()

        return PyTreeDict(actor_loss=actor_loss, entropy=entropy)

    def critic_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        obs = sample_batch.obs
        next_obs = sample_batch.extras.env_extras.last_obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )

        discounts = self.discount * (1 - sample_batch.extras.env_extras.termination)

        alpha = jnp.exp(agent_state.params.log_alpha)

        # [B, 2]
        q_t = self.critic_network.apply(
            agent_state.params.critic_params, obs, sample_batch.actions
        )

        actions_dist_t_plus_1 = get_tanh_norm_dist(
            *jnp.split(
                self.actor_network.apply(agent_state.params.actor_params, next_obs),
                2,
                axis=-1,
            )
        )
        actions_t_plus_1 = actions_dist_t_plus_1.sample(seed=key)
        actions_logp_t_plus_1 = actions_dist_t_plus_1.log_prob(actions_t_plus_1)
        # [B, 2]
        q_t_plus_1 = self.critic_network.apply(
            agent_state.params.target_critic_params, next_obs, actions_t_plus_1
        )
        q_target = sample_batch.rewards * self.reward_scale + discounts * (
            jnp.min(q_t_plus_1, axis=-1) - alpha * actions_logp_t_plus_1
        )
        q_target = jnp.repeat(q_target[..., None], 2, axis=-1)

        q_loss = optax.squared_error(q_t, q_target).sum(-1).mean()
        return PyTreeDict(critic_loss=q_loss)


class SACDiscreteAgent(Agent):
    critic_hidden_layer_sizes: tuple[int] = (256, 256)
    actor_hidden_layer_sizes: tuple[int] = (256, 256)
    alpha: float = 0.2
    adaptive_alpha: bool = False
    discount: float = 0.99
    # reward_scale: float = 1.0
    normalize_obs: bool = False
    critic_network: nn.Module = pytree_field(lazy_init=True)
    actor_network: nn.Module = pytree_field(lazy_init=True)
    obs_preprocessor: Any = pytree_field(lazy_init=True, pytree_node=False)


class SACWorkflow(OffPolicyWorkflowTemplate):
    @classmethod
    def name(cls):
        return "SAC"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            env_name=config.env.env_name,
            env_type=config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
        )

        assert isinstance(
            env.action_space, Box
        ), "Only continue action space is supported."

        agent = SACAgent(
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            init_alpha=config.alpha,
            discount=config.discount,
            reward_scale=config.reward_scale,
            normalize_obs=config.normalize_obs,
        )

        # TODO: use different lr for critic and actor
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

        replay_buffer = flashbax.make_item_buffer(
            max_length=config.replay_buffer_capacity,
            min_length=max(config.batch_size, config.learning_start_timesteps),
            sample_batch_size=config.batch_size,
            add_batches=True,
        )

        eval_env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env, agent=agent, max_episode_steps=config.env.max_episode_steps
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
            dict(
                actor=self.optimizer.init(agent_state.params.actor_params),
                critic=self.optimizer.init(agent_state.params.critic_params),
            )
        )
        if self.config.adaptive_alpha:
            opt_state = opt_state.replace(
                alpha=self.optimizer.init(agent_state.params.log_alpha)
            )

        return agent_state, opt_state

    def step(self, state: State) -> tuple[TrainMetric, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        # the trajectory [T, B, ...]
        trajectory, env_state = rollout(
            env_fn=self.env.step,
            action_fn=self.agent.compute_actions,
            env_state=state.env_state,
            agent_state=state.agent_state,
            key=rollout_key,
            rollout_length=self.config.rollout_length,
            env_extra_fields=("last_obs", "termination"),
        )

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

            loss = self.config.loss_weights.critic_loss * loss_dict.critic_loss
            return loss, loss_dict

        def actor_loss_fn(agent_state, sample_batch, key):
            loss_dict = self.agent.actor_loss(agent_state, sample_batch, key)

            loss = self.config.loss_weights.actor_loss * loss_dict.actor_loss
            return loss, loss_dict

        def alpha_loss_fn(agent_state, sample_batch, key):
            loss_dict = self.agent.alpha_loss(agent_state, sample_batch, key)

            loss = self.config.loss_weights.alpha_loss * loss_dict.alpha_loss
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

        alpha_update_fn = agent_gradient_update(
            alpha_loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True,
            attach_fn=lambda agent_state, log_alpha: agent_state.replace(
                params=agent_state.params.replace(log_alpha=log_alpha)
            ),
            detach_fn=lambda agent_state: agent_state.params.log_alpha,
        )

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            critic_opt_state = opt_state.critic
            actor_opt_state = opt_state.actor

            key, critic_key, actor_key, alpha_key, rb_key = jax.random.split(key, num=5)

            if self.config.actor_update_interval - 1 > 0:

                def _sample_and_update_critic_fn(carry, unused_t):
                    key, agent_state, critic_opt_state = carry

                    key, rb_key, critic_key = jax.random.split(key, num=3)
                    # it's safe to use read-only replay_buffer_state here.
                    sample_batch = self.replay_buffer.sample(
                        replay_buffer_state, rb_key
                    ).experience

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

            sample_batch = self.replay_buffer.sample(
                replay_buffer_state, rb_key
            ).experience

            (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
                critic_update_fn(
                    critic_opt_state, agent_state, sample_batch, critic_key
                )
            )

            (actor_loss, actor_loss_dict), agent_state, actor_opt_state = (
                actor_update_fn(actor_opt_state, agent_state, sample_batch, actor_key)
            )

            opt_state = opt_state.replace(
                actor=actor_opt_state, critic=critic_opt_state
            )

            if self.config.adaptive_alpha:
                # we follow the update order of the official implementation:
                # critic -> actor -> alpha
                alpha_opt_state = opt_state.alpha
                (alpha_loss, alpha_loss_dict), agent_state, alpha_opt_state = (
                    alpha_update_fn(
                        alpha_opt_state, agent_state, sample_batch, alpha_key
                    )
                )
                opt_state = opt_state.replace(alpha=alpha_opt_state)
                res = (
                    critic_loss,
                    actor_loss,
                    alpha_loss,
                    critic_loss_dict,
                    actor_loss_dict,
                    alpha_loss_dict,
                )
            else:
                res = (critic_loss, actor_loss, critic_loss_dict, actor_loss_dict)

            target_critic_params = soft_target_update(
                agent_state.params.target_critic_params,
                agent_state.params.critic_params,
                self.config.tau,
            )
            agent_state = agent_state.replace(
                params=agent_state.params.replace(
                    target_critic_params=target_critic_params
                )
            )

            return (key, agent_state, opt_state), res

        if self.config.adaptive_alpha:
            (
                (_, agent_state, opt_state),
                (
                    critic_loss,
                    actor_loss,
                    alpha_loss,
                    critic_loss_dict,
                    actor_loss_dict,
                    alpha_loss_dict,
                ),
            ) = scan_and_mean(
                _sample_and_update_fn,
                (learn_key, agent_state, state.opt_state),
                (),
                length=self.config.num_updates_per_iter,
            )
            train_metrics = SACTrainMetric(
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                alpha_loss=alpha_loss,
                raw_loss_dict=PyTreeDict(
                    {**critic_loss_dict, **actor_loss_dict, **alpha_loss_dict}
                ),
            ).all_reduce(pmap_axis_name=self.pmap_axis_name)
        else:
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
            train_metrics = SACTrainMetric(
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
            ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        # calculate the number of timestep
        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.pmap_axis_name,
        )

        # iterations is the number of updates of the agent
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
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
