from chex._src.pytypes import PRNGKey
import jax
import jax.numpy as jnp
from flax import struct
import orbax.checkpoint as ocp

from .agent import Agent, AgentState
from .random_agent import RandomAgent, EMPTY_RANDOM_AGENT_STATE
from evorl.networks import make_q_network, make_policy_network
from evorl.workflows import OffPolicyRLWorkflow
from evorl.envs import create_env, Box
from evorl.sample_batch import SampleBatch
from evorl.evaluator import Evaluator
from evorl.utils import running_statistics
from evorl.rollout import rollout
from evox import State
from evorl.distributed import split_key_to_devices, tree_unpmap, psum, agent_gradient_update, tree_pmean
from evorl.utils.jax_utils import tree_stop_gradient, scan_and_mean, tree_last
from evorl.utils.toolkits import average_episode_discount_return, soft_target_update, flatten_rollout_trajectory
from evorl.metrics import MetricBase, metricfield
from evorl.distribution import get_tanh_norm_dist
from omegaconf import DictConfig
from typing import Tuple, Any, Callable, Optional
import optax
import chex
from evorl.metrics import TrainMetric, WorkflowMetric
import flashbax
from evorl.types import (
    LossDict,
    Action,
    Params,
    PolicyExtraInfo,
    PyTreeDict,
    PyTreeData,
    pytree_field,
    MISSING_REWARD,
)
import logging
import flax.linen as nn
import math

logger = logging.getLogger(__name__)
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


class SACTrainMetric(MetricBase):
    critic_loss: chex.Array
    actor_loss: chex.Array
    alpha_loss: Optional[chex.Array] = None
    raw_loss_dict: LossDict = metricfield(
        default_factory=PyTreeDict, reduce_fn=tree_pmean)


class SACNetworkParams(PyTreeData):
    critic_params: Params
    target_critic_params: Params
    actor_params: Params
    log_alpha: Params


class SACAgent(Agent):
    critic_hidden_layer_sizes: Tuple[int] = (256, 256)
    actor_hidden_layer_sizes: Tuple[int] = (256, 256)
    init_alpha: float = 1.0
    discount: float = 0.99
    reward_scale: float = 1.0
    normalize_obs: bool = False
    critic_network: nn.Module = pytree_field(lazy_init=True)
    actor_network: nn.Module = pytree_field(lazy_init=True)
    obs_preprocessor: Any = pytree_field(lazy_init=True, pytree_node=False)

    def init(self, key: chex.PRNGKey) -> AgentState:
        obs_size = self.obs_space.shape[0]
        action_size = self.action_space.shape[0]

        key, critic_key, actor_key, obs_preprocessor_key = jax.random.split(
            key, num=4)

        critic_network, critic_init_fn = make_q_network(
            obs_size=obs_size,
            action_size=action_size,
            n_stack=2,
            hidden_layer_sizes=self.critic_hidden_layer_sizes,
        )

        critic_params = critic_init_fn(critic_key)
        target_critic_params = critic_params

        actor_network, actor_init_fn = make_policy_network(
            action_size=action_size*2,  # mean+std
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
            log_alpha=log_alpha
        )

        if self.normalize_obs:
            obs_preprocessor = running_statistics.normalize
            self.set_frozen_attr('obs_preprocessor', obs_preprocessor)
            dummy_obs = self.obs_space.sample(obs_preprocessor_key)
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> Tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        raw_actions = self.actor_network.apply(
            agent_state.params.actor_params, obs)
        action_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        action = action_dist.sample(seed=key)
        return action, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> Tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        raw_actions = self.actor_network.apply(
            agent_state.params.actor_params, obs)
        action_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        action = action_dist.mode()
        return jax.lax.stop_gradient(action), PyTreeDict()

    def alpha_loss(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        raw_actions = self.actor_network.apply(
            agent_state.params.actor_params, obs)
        actions_dist = get_tanh_norm_dist(
            *jnp.split(raw_actions, 2, axis=-1))
        actions = actions_dist.sample(seed=key)
        actions_logp = actions_dist.log_prob(actions)

        entropy_target = - \
            jnp.prod(jnp.array(self.action_space.shape, dtype=jnp.float32))
        # official impl:
        alpha = jnp.exp(agent_state.params.log_alpha)
        alpha_loss = jnp.mean(- alpha *
                              jax.lax.stop_gradient(actions_logp + entropy_target))

        # another impl: see stable-baselines3/issues/36
        # alpha_loss = (- agent_state.params.log_alpha *
        #               jax.lax.stop_gradient(actions_logp + entropy_target)).mean()

        return PyTreeDict(
            alpha_loss=alpha_loss,
            log_alpha=agent_state.params.log_alpha,
            alpha=alpha
        )

    def actor_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        actor_key, entropy_key = jax.random.split(key, 2)
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        alpha = jnp.exp(agent_state.params.log_alpha)

        raw_actions = self.actor_network.apply(
            agent_state.params.actor_params, obs)
        actions_dist = get_tanh_norm_dist(
            *jnp.split(raw_actions, 2, axis=-1))
        actions = actions_dist.sample(seed=actor_key)
        actions_logp = actions_dist.log_prob(actions)

        # [B, 2]
        q_values = self.critic_network.apply(
            agent_state.params.critic_params, obs, actions)
        min_q = jnp.min(q_values, axis=-1)
        actor_loss = jnp.mean(alpha * actions_logp - min_q)
        entropy = actions_dist.entropy(seed=entropy_key).mean()

        return PyTreeDict(
            actor_loss=actor_loss,
            entropy=entropy
        )

    def critic_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        obs = sample_batch.obs
        next_obs = sample_batch.extras.env_extras.last_obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state)

        discounts = self.discount * \
            (1-sample_batch.extras.env_extras.termination)

        alpha = jnp.exp(agent_state.params.log_alpha)

        # [B, 2]
        q_t = self.critic_network.apply(
            agent_state.params.critic_params, obs, sample_batch.actions
        )

        actions_dist_t_plus_1 = get_tanh_norm_dist(
            *jnp.split(
                self.actor_network.apply(
                    agent_state.params.actor_params,
                    next_obs
                ), 2, axis=-1
            ))
        actions_t_plus_1 = actions_dist_t_plus_1.sample(seed=key)
        actions_logp_t_plus_1 = actions_dist_t_plus_1.log_prob(
            actions_t_plus_1)
        # [B, 2]
        q_t_plus_1 = self.critic_network.apply(
            agent_state.params.target_critic_params,
            next_obs, actions_t_plus_1
        )
        q_target = sample_batch.rewards * self.reward_scale + \
            discounts * (jnp.min(q_t_plus_1, axis=-1) -
                         alpha * actions_logp_t_plus_1)
        q_target = jnp.repeat(q_target[..., None], 2, axis=-1)

        q_loss = optax.squared_error(q_t, q_target).mean()
        return PyTreeDict(critic_loss=q_loss)


class SACDiscreteAgent(Agent):
    critic_hidden_layer_sizes: Tuple[int] = (256, 256)
    actor_hidden_layer_sizes: Tuple[int] = (256, 256)
    alpha: float = 0.2
    adaptive_alpha: bool = False
    discount: float = 0.99
    # reward_scale: float = 1.0
    normalize_obs: bool = False
    critic_network: nn.Module = pytree_field(lazy_init=True)
    actor_network: nn.Module = pytree_field(lazy_init=True)
    obs_preprocessor: Any = pytree_field(lazy_init=True, pytree_node=False)


class SACWorkflow(OffPolicyRLWorkflow):
    @classmethod
    def name(cls):
        return "SAC"

    @staticmethod
    def _rescale_config(config) -> None:
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
        if config.replay_buffer_capacity % num_devices != 0:
            logger.warning(
                f"replay_buffer_capacity({config.replay_buffer_capacity}) cannot be divided by num_devices({num_devices}), "
                f"rescale replay_buffer_capacity to {config.replay_buffer_capacity // num_devices}"
            )
        if config.random_timesteps % num_devices != 0:
            logger.warning(
                f"random_timesteps({config.random_timesteps}) cannot be divided by num_devices({num_devices}), "
                f"rescale random_timesteps to {config.random_timesteps // num_devices}"
            )
        if config.learning_start_timesteps % num_devices != 0:
            logger.warning(
                f"learning_start_timesteps({config.learning_start_timesteps}) cannot be divided by num_devices({num_devices}), "
                f"rescale learning_start_timesteps to {config.learning_start_timesteps // num_devices}"
            )

        config.num_envs = config.num_envs // num_devices
        config.num_eval_envs = config.num_eval_envs // num_devices
        config.replay_buffer_capacity = config.replay_buffer_capacity // num_devices
        config.random_timesteps = config.random_timesteps // num_devices
        config.learning_start_timesteps = config.learning_start_timesteps // num_devices

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            env_name=config.env.env_name,
            env_type=config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset=True,
        )

        assert isinstance(
            env.action_space, Box
        ), "Only continue action space is supported."

        agent = SACAgent(
            action_space=env.action_space,
            obs_space=env.obs_space,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            init_alpha=config.alpha,
            discount=config.discount,
            reward_scale=config.reward_scale,
            normalize_obs=config.normalize_obs,
        )

        # TODO: use different lr for critic and actor
        if (config.optimizer.grad_clip_norm is not None and
                config.optimizer.grad_clip_norm > 0):
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.optimizer.grad_clip_norm),
                optax.adam(config.optimizer.lr)
            )
        else:
            optimizer = optax.adam(config.optimizer.lr)

        replay_buffer = flashbax.make_item_buffer(
            max_length=config.replay_buffer_capacity,
            min_length=config.learning_start_timesteps,
            sample_batch_size=config.batch_size,
            add_batches=True,
        )

        eval_env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset=False,
        )

        evaluator = Evaluator(
            env=eval_env,
            agent=agent,
            max_episode_steps=config.env.max_episode_steps
        )

        return cls(
            env,
            agent,
            optimizer,
            evaluator,
            replay_buffer,
            config,
        )

    def _setup_agent_and_optimizer(self, key: chex.PRNGKey) -> tuple[AgentState, chex.ArrayTree]:
        agent_state = self.agent.init(key)
        opt_state = PyTreeDict(dict(
            actor=self.optimizer.init(agent_state.params.actor_params),
            critic=self.optimizer.init(agent_state.params.critic_params),
        ))
        if self.config.adaptive_alpha:
            opt_state = opt_state.replace(
                alpha=self.optimizer.init(agent_state.params.log_alpha)
            )

        return agent_state, opt_state

    def _setup_replaybuffer(self, key: chex.PRNGKey) -> chex.ArrayTree:
        action_space = self.env.action_space
        obs_space = self.env.obs_space

        # create dummy data to initialize the replay buffer
        dummy_action = jnp.zeros(action_space.shape)
        dummy_obs = jnp.zeros(obs_space.shape)

        dummy_reward = jnp.zeros(())
        dummy_done = jnp.zeros(())

        dummy_sample_batch = SampleBatch(
            obs=dummy_obs,
            actions=dummy_action,
            rewards=dummy_reward,
            # next_obs=dummy_obs,
            # dones=dummy_done,
            extras=PyTreeDict(
                policy_extras=PyTreeDict(),
                env_extras=PyTreeDict(
                    {"last_obs": dummy_obs, "termination": dummy_done}
                ),
            ),
        )
        replay_buffer_state = self.replay_buffer.init(dummy_sample_batch)

        return replay_buffer_state

    def _postsetup_replaybuffer(self, state: State) -> State:
        action_space = self.env.action_space
        obs_space = self.env.obs_space
        config = self.config
        replay_buffer_state = state.replay_buffer_state
        agent_state = state.agent_state

        # ==== fill random transitions ====
        key, env_key, rollout_key = jax.random.split(state.key, 3)
        random_agent = RandomAgent(
            action_space=action_space, obs_space=obs_space)

        # Note: in multi-devices mode, this method is running in pmap, and
        # config.num_envs = config.num_envs // num_devices
        # config.random_timesteps = config.random_timesteps // num_devices

        rollout_length = config.random_timesteps // config.num_envs
        env_state = self.env.reset(env_key)

        trajectory, env_state = rollout(
            env_fn=self.env.step,
            action_fn=random_agent.compute_actions,
            env_state=env_state,
            agent_state=EMPTY_RANDOM_AGENT_STATE,
            key=rollout_key,
            rollout_length=rollout_length,
            env_extra_fields=("last_obs", "termination"),
        )

        # [T, B, ...] -> [T*B, ...]
        trajectory = clean_trajectory(trajectory)
        trajectory = flatten_rollout_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    pmap_axis_name=self.pmap_axis_name,
                )
            )

        replay_buffer_state = self.replay_buffer.add(
            replay_buffer_state, trajectory)

        rollout_timesteps = rollout_length*config.num_envs
        sampled_timesteps = psum(
            jnp.uint32(rollout_timesteps), axis_name=self.pmap_axis_name
        )

        # ==== fill tansition state from init agent ====
        rollout_length = math.ceil((config.learning_start_timesteps -
                                    rollout_timesteps) / config.num_envs)
        key, env_key, rollout_key = jax.random.split(key, 3)

        env_state = self.env.reset(env_key)
        trajectory, env_state = rollout(
            env_fn=self.env.step,
            action_fn=self.agent.compute_actions,
            env_state=env_state,
            agent_state=state.agent_state,
            key=rollout_key,
            rollout_length=rollout_length,
            env_extra_fields=("last_obs", "termination"),
        )

        trajectory = clean_trajectory(trajectory)
        trajectory = flatten_rollout_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        replay_buffer_state = self.replay_buffer.add(
            replay_buffer_state, trajectory)

        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    pmap_axis_name=self.pmap_axis_name,
                )
            )

        rollout_timesteps = rollout_length*config.num_envs
        sampled_timesteps += psum(
            jnp.uint32(rollout_timesteps), axis_name=self.pmap_axis_name
        )

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps+sampled_timesteps,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return state.replace(
            key=key,
            metrics=workflow_metrics,
            replay_buffer_state=replay_buffer_state
        )

    def step(self, state: State) -> Tuple[TrainMetric, State]:

        key, rollout_key, learn_key = jax.random.split(
            state.key, num=3)

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
            loss_dict = self.agent.critic_loss(
                agent_state, sample_batch, key)

            loss = self.config.loss_weights.critic_loss *\
                loss_dict.critic_loss
            return loss, loss_dict

        def actor_loss_fn(agent_state, sample_batch, key):
            loss_dict = self.agent.actor_loss(
                agent_state, sample_batch, key)

            loss = self.config.loss_weights.actor_loss * \
                loss_dict.actor_loss
            return loss, loss_dict

        def alpha_loss_fn(agent_state, sample_batch, key):
            loss_dict = self.agent.alpha_loss(
                agent_state, sample_batch, key)

            loss = self.config.loss_weights.alpha_loss * \
                loss_dict.alpha_loss
            return loss, loss_dict

        critic_update_fn = agent_gradient_update(
            critic_loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True,
            attach_fn=lambda agent_state, critic_params: agent_state.replace(
                params=agent_state.params.replace(critic_params=critic_params)
            ),
            detach_fn=lambda agent_state: agent_state.params.critic_params
        )

        actor_update_fn = agent_gradient_update(
            actor_loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True,
            attach_fn=lambda agent_state, actor_params: agent_state.replace(
                params=agent_state.params.replace(actor_params=actor_params)
            ),
            detach_fn=lambda agent_state: agent_state.params.actor_params
        )

        alpha_update_fn = agent_gradient_update(
            alpha_loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True,
            attach_fn=lambda agent_state, log_alpha: agent_state.replace(
                params=agent_state.params.replace(log_alpha=log_alpha)
            ),
            detach_fn=lambda agent_state: agent_state.params.log_alpha
        )

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            critic_opt_state = opt_state.critic
            actor_opt_state = opt_state.actor

            key, critic_key, actor_key, alpha_key, rb_key = jax.random.split(
                key, num=5)

            if self.config.actor_update_interval-1 > 0:
                def _sample_and_update_critic_fn(carry, unused_t):
                    key, agent_state, critic_opt_state = carry

                    key, rb_key, critic_key = jax.random.split(key, num=3)
                    # it's safe to use read-only replay_buffer_state here.
                    sampled_batch = self.replay_buffer.sample(
                        replay_buffer_state, rb_key).experience

                    (critic_loss, critic_loss_dict), agent_state, critic_opt_state = critic_update_fn(
                        critic_opt_state,
                        agent_state,
                        sampled_batch,
                        critic_key
                    )

                    return (key, agent_state, critic_opt_state), None

                key, critic_multiple_update_key = jax.random.split(key)

                (_, agent_state, critic_opt_state), _ = jax.lax.scan(
                    _sample_and_update_critic_fn,
                    (critic_multiple_update_key, agent_state, critic_opt_state),
                    (),
                    length=self.config.actor_update_interval-1
                )

            sampled_batch = self.replay_buffer.sample(
                replay_buffer_state, rb_key).experience

            (critic_loss, critic_loss_dict), agent_state, critic_opt_state = critic_update_fn(
                critic_opt_state,
                agent_state,
                sampled_batch,
                critic_key
            )

            (actor_loss, actor_loss_dict), agent_state, actor_opt_state = actor_update_fn(
                actor_opt_state,
                agent_state,
                sampled_batch,
                actor_key
            )

            opt_state = opt_state.replace(
                actor=actor_opt_state,
                critic=critic_opt_state
            )

            if self.config.adaptive_alpha:
                alpha_opt_state = opt_state.alpha
                (alpha_loss, alpha_loss_dict), agent_state, alpha_opt_state = alpha_update_fn(
                    alpha_opt_state,
                    agent_state,
                    sampled_batch,
                    alpha_key
                )
                opt_state = opt_state.replace(alpha=alpha_opt_state)
                res = (critic_loss, actor_loss, alpha_loss,
                       critic_loss_dict, actor_loss_dict, alpha_loss_dict)
            else:
                res = (critic_loss, actor_loss,
                       critic_loss_dict, actor_loss_dict)

            target_critc_params = soft_target_update(
                agent_state.params.target_critic_params,
                agent_state.params.critic_params,
                self.config.tau,
            )
            agent_state = agent_state.replace(
                params=agent_state.params.replace(
                    target_critic_params=target_critc_params
                )
            )

            return (key, agent_state, opt_state), res

        if self.config.adaptive_alpha:
            (_, agent_state, opt_state), \
                (critic_loss, actor_loss, alpha_loss,
                 critic_loss_dict, actor_loss_dict, alpha_loss_dict) = scan_and_mean(
                _sample_and_update_fn,
                (learn_key, agent_state, state.opt_state),
                (),
                length=self.config.num_updates_per_iter
            )
            train_metrics = SACTrainMetric(
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                alpha_loss=alpha_loss,
                raw_loss_dict=PyTreeDict(
                    {**critic_loss_dict, **actor_loss_dict, **alpha_loss_dict}),
            ).all_reduce(pmap_axis_name=self.pmap_axis_name)
        else:
            (_, agent_state, opt_state), \
                (critic_loss, actor_loss, critic_loss_dict, actor_loss_dict) = scan_and_mean(
                _sample_and_update_fn,
                (learn_key, agent_state, state.opt_state),
                (),
                length=self.config.num_updates_per_iter
            )
            train_metrics = SACTrainMetric(
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                raw_loss_dict=PyTreeDict(
                    {**critic_loss_dict, **actor_loss_dict}),
            ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        # calculate the numbner of timestep
        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.pmap_axis_name
        )

        # iterations is the number of updates of the agent
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps+sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            replay_buffer_state=replay_buffer_state,
            opt_state=opt_state
        )

    def _multi_steps(self, state):
        def _step(state, _):
            train_metrics, state = self.step(state)
            return state, train_metrics

        state, train_metrics = jax.lax.scan(
            _step, state, (), length=self.config.fold_iters)
        train_metrics = tree_last(train_metrics)
        return train_metrics, state

    def learn(self, state: State) -> State:
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        sampled_timesteps = tree_unpmap(
            state.metrics.sampled_timesteps).tolist()
        num_iters = math.ceil(
            (self.config.total_timesteps-sampled_timesteps) /
            (one_step_timesteps*self.config.fold_iters)
        )

        for i in range(num_iters):
            train_metrics, state = self._multi_steps(state)
            workflow_metrics = state.metrics

            # current iteration
            iterations = tree_unpmap(
                state.metrics.iterations, self.pmap_axis_name).tolist()
            train_metrics = tree_unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = tree_unpmap(
                workflow_metrics, self.pmap_axis_name
            )
            self.recorder.write(
                train_metrics.to_local_dict(), iterations)
            self.recorder.write(
                workflow_metrics.to_local_dict(), iterations)

            if iterations % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)
                eval_metrics = tree_unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write(
                    {"eval": eval_metrics.to_local_dict()}, iterations)

            saved_state = tree_unpmap(state, self.pmap_axis_name)
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)
            self.checkpoint_manager.save(
                iterations,
                args=ocp.args.StandardSave(saved_state),
            )

        return state

    @classmethod
    def enable_jit(cls) -> None:
        super().enable_jit()
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,))
        cls._multi_steps = jax.jit(cls._multi_steps, static_argnums=(0,))

def skip_replay_buffer_state(state: State) -> State:
    return state.replace(replay_buffer_state=None)

def clean_trajectory(trajectory):
    """
    clean the trajectory to make it suitable for the replay buffer
    """
    return trajectory.replace(
        next_obs=None,
        dones=None,
    )
