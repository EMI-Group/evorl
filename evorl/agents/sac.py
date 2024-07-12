from chex._src.pytypes import PRNGKey
import jax
import jax.numpy as jnp
from flax import struct
import orbax.checkpoint as ocp

from .agent import Agent, AgentState
from .random_agent import RandomAgent
from evorl.networks import make_q_network, make_policy_network
from evorl.workflows import OffPolicyRLWorkflow
from evorl.envs import create_env, Box
from evorl.sample_batch import SampleBatch
from evorl.evaluator import Evaluator
from evorl.utils import running_statistics
from evorl.rollout import rollout
from evox import State
from evorl.distributed import split_key_to_devices, tree_unpmap, psum, agent_gradient_update
from evorl.utils.toolkits import average_episode_discount_return, soft_target_update, flatten_rollout_trajectory
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
    pytree_field,
    MISSING_REWARD,
)
import logging
import flax.linen as nn
import math

logger = logging.getLogger(__name__)
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@struct.dataclass
class SACNetworkParams:
    critic_params: Params
    target_critic_params: Params
    actor_params: Params
    alpha_params: Params

class SACAgent(Agent):
    critic_hidden_layer_sizes: Tuple[int] = (256, 256)
    actor_hidden_layer_sizes: Tuple[int] = (256, 256)
    normalize_obs: bool = False
    critic_network: nn.Module = pytree_field(lazy_init=True)
    actor_network: nn.Module = pytree_field(lazy_init=True)
    obs_preprocessor: Any = pytree_field(lazy_init=True, pytree_node=False)
    alpha: float = 0.2
    adaptive_alpha: bool = False
    discount: float = 0.99
    reward_scale: float = 1.0

    def init(self, key: chex.PRNGKey) -> AgentState:
        obs_size = self.obs_space.shape[0]
        action_size = self.action_space.shape[0]

        key, critic_key, actor_key, obs_preprocessor_key = jax.random.split(key, num=4)

        critic_network, critic_init_fn = make_q_network(
            obs_size=obs_size,
            action_size=action_size,
            hidden_layer_sizes=self.critic_hidden_layer_sizes,
            n_critics=2,
        )

        critic_params = critic_init_fn(critic_key)
        target_critic_params = critic_params

        actor_network, actor_init_fn = make_policy_network(
            action_size=2*action_size,
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes,
        )

        actor_params = actor_init_fn(actor_key)

        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
            
        self.set_frozen_attr("critic_network", critic_network)
        self.set_frozen_attr("actor_network", actor_network)

        params_state = SACNetworkParams(
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            actor_params=actor_params,
            alpha_params=log_alpha
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

        raw_actions = self.actor_network.apply(agent_state.params.actor_params, obs)
        action_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        action = action_dist.sample(seed=key)
        return jax.lax.stop_gradient(action), PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> Tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        raw_actions = self.actor_network.apply(agent_state.params.actor_params, obs)
        action_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        action = action_dist.mode()
        return jax.lax.stop_gradient(action), PyTreeDict()
    
    def alpha_loss(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        dist = get_tanh_norm_dist(*jnp.split(self.actor_network.apply(agent_state.params.actor_params, obs), 2, axis=-1))
        actions = dist.sample(seed=key)
        log_prob = dist.log_prob(actions)
        alpha = jnp.exp(agent_state.params.alpha_params)
        target_entropy = -0.5 * self.action_space.shape[0]
        alpha_loss = alpha * jax.lax.stop_gradient(- log_prob - target_entropy).mean()
        return PyTreeDict(alpha_loss=alpha_loss, alpha=alpha)

    def actor_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        actor_key, entropy_key = jax.random.split(key, 2)
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)
            
        alpha = self.alpha
        if self.adaptive_alpha:
            alpha = jnp.exp(agent_state.params.alpha_params)

        dist = get_tanh_norm_dist(*jnp.split(self.actor_network.apply(agent_state.params.actor_params, obs), 2, axis=-1))
        actions = dist.sample(seed=actor_key)
        log_prob = dist.log_prob(actions)
        q_values = self.critic_network.apply(agent_state.params.critic_params, obs, actions)
        min_q = jnp.min(q_values, axis=-1)
        actor_loss = jnp.mean(alpha * log_prob - min_q)
        entropy_loss = dist.entropy(seed=entropy_key).mean()

        return PyTreeDict(actor_loss=actor_loss, entropy_loss=entropy_loss)
    
    def critic_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)
            
        alpha = self.alpha
        if self.adaptive_alpha:
            alpha = jnp.exp(agent_state.params.alpha_params)
        
        old_q_values = self.critic_network.apply(
            agent_state.params.critic_params, obs, sample_batch.actions
        )
        next_dist = get_tanh_norm_dist(*jnp.split(self.actor_network.apply(agent_state.params.actor_params, sample_batch.next_obs), 2, axis=-1))
        next_actions = next_dist.sample(seed=key)
        next_log_prob = next_dist.log_prob(next_actions)
        next_q_values = self.critic_network.apply(
            agent_state.params.target_critic_params, sample_batch.next_obs, next_actions
        )
        next_v = jnp.min(next_q_values, axis=-1) - alpha * next_log_prob
        target_q = jax.lax.stop_gradient(sample_batch.rewards * self.reward_scale + self.discount * (1.0 - sample_batch.dones) * next_v)
        q_error = old_q_values - jnp.expand_dims(target_q, axis=-1)
        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return PyTreeDict(critic_loss=q_loss)
    
    def loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        pass

    def compute_values(
        self, agent_state: AgentState, sample_batch: SampleBatch
    ) -> chex.Array:
        obs = sample_batch.obs
        actions = sample_batch.actions
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        return self.critic_network.apply(
            agent_state.params.value_params, obs, actions
        )


class SACWorkflow(OffPolicyRLWorkflow):
    @classmethod
    def name(cls):
        return "SAC"

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
            normalize_obs=config.normalize_obs,
            alpha=config.alpha,
            adaptive_alpha=config.adaptive_alpha,
            discount=config.discount,
            reward_scale=config.reward_scale,
        )

        if (config.optimizer.grad_clip_norm is not None and
                config.optimizer.grad_clip_norm > 0):
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.optimizer.grad_clip_norm),
                optax.adam(config.optimizer.lr)
            )
        else:
            optimizer = optax.adam(config.optimizer.lr)

        replay_buffer = flashbax.make_item_buffer(
            max_length=config.replay_buffer.capacity,
            min_length=config.replay_buffer.min_size,
            sample_batch_size=config.replay_buffer.sample_batch_size,
            add_batches=True
        )

        def _replay_buffer_init_fn(replay_buffer, key):
            dummy_action = jnp.zeros(env.action_space.shape)
            dummy_obs = jnp.zeros(env.obs_space.shape)
            dummy_reward = jnp.zeros(())
            dummy_done = jnp.zeros(())
            dummy_episode_return = jnp.zeros(())
            dummy_next_obs = dummy_obs

            dummy_sample_batch = SampleBatch(
                obs=dummy_obs,
                actions=dummy_action,
                rewards=dummy_reward,
                next_obs=dummy_next_obs,
                dones=dummy_done,
                extras=PyTreeDict(
                    policy_extras=PyTreeDict(),
                    env_extras=PyTreeDict(
                        {"last_obs": dummy_obs, "truncation": dummy_done, 'episode_return': dummy_episode_return}
                    ),
                ),
            )
            replay_buffer_state = replay_buffer.init(dummy_sample_batch)

            return replay_buffer_state

        eval_env = create_env(
            env_name=config.env.env_name,
            env_type=config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset=False
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
            _replay_buffer_init_fn,
            config,
        )
    
    def setup(self, key: chex.PRNGKey) -> State:
        self.recorder.init()
        key, agent_key, env_key, buffer_key = jax.random.split(key, 4)

        agent_state = self.agent.init(agent_key)

        workflow_metrics = self._setup_workflow_metrics()
        
        critic_opt_state = self.optimizer.init(agent_state.params.critic_params)
        actor_opt_state = self.optimizer.init(agent_state.params.actor_params)
        alpha_opt_state = self.optimizer.init(agent_state.params.alpha_params)


        replay_buffer_state = self._init_replay_buffer(
            self.replay_buffer, buffer_key)

        if self.enable_multi_devices:
            (
                workflow_metrics,
                agent_state,
                critic_opt_state,
                actor_opt_state,
                alpha_opt_state,
                replay_buffer_state,
            ) = jax.device_put_replicated(
                (
                    workflow_metrics,
                    agent_state,
                    critic_opt_state,
                    actor_opt_state,
                    alpha_opt_state,
                    replay_buffer_state,
                ),
                self.devices,
            )

            # key and env_state should be different over devices
            key = split_key_to_devices(key, self.devices)

            env_key = split_key_to_devices(env_key, self.devices)
            env_state = jax.pmap(self.env.reset, axis_name=self.pmap_axis_name)(env_key)
        else:
            env_state = self.env.reset(env_key)

        return State(
            key=key,
            metrics=workflow_metrics,
            replay_buffer_state=replay_buffer_state,
            agent_state=agent_state,
            env_state=env_state,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            alpha_opt_state=alpha_opt_state,
        )

    # def setup(self, key: chex.PRNGKey) -> State:
    #     key, agent_key, env_key, buffer_key = jax.random.split(key, 4)

    #     agent_state = self.agent.init(agent_key)

    #     workflow_metrics = self._setup_workflow_metrics()

    #     critic_opt_state = self.optimizer.init(agent_state.params.critic_params)
    #     actor_opt_state = self.optimizer.init(agent_state.params.actor_params)
    #     alpha_opt_state = self.optimizer.init(agent_state.params.alpha_params)

    #     replay_buffer_state = self._init_replay_buffer(self.replay_buffer, buffer_key)

    #     if self.enable_multi_devices:
    #         (
    #             workflow_metrics,
    #             agent_state,
    #             critic_opt_state,
    #             actor_opt_state,
    #             alpha_opt_state,
    #             replay_buffer_state,
    #         ) = jax.device_put_replicated(
    #             (
    #                 workflow_metrics,
    #                 agent_state,
    #                 critic_opt_state,
    #                 actor_opt_state,
    #                 alpha_opt_state,
    #                 replay_buffer_state,
    #             ),
    #             self.devices,
    #         )

    #         # key and env_state should be different over devices
    #         key = split_key_to_devices(key, self.devices)

    #         env_key = split_key_to_devices(env_key, self.devices)
    #         env_state = jax.pmap(self.env.reset, axis_name=self.pmap_axis_name)(env_key)
    #     else:
    #         env_state = self.env.reset(env_key)

    #     return State(
    #         key=key,
    #         metrics=workflow_metrics,
    #         replay_buffer_state=replay_buffer_state,
    #         agent_state=agent_state,
    #         env_state=env_state,
    #         actor_opt_state=actor_opt_state,
    #         critic_opt_state=critic_opt_state,
    #         alpha_opt_state=alpha_opt_state,
    #     )

    def step(self, state: State) -> Tuple[TrainMetric, State]:

        key, rollout_key, learn_key, buffer_key = jax.random.split(state.key, num=4)

        def random_agent_rollout_fn(state:State):
            random_agent = RandomAgent(self.env.action_space, self.env.obs_space)
            random_agent_state = AgentState(params={})
            env_state, trajectory = rollout(
                env=self.env,
                agent=random_agent,
                env_state=state.env_state,
                agent_state=random_agent_state,
                key=rollout_key,
                rollout_length=self.config.rollout_length,
                env_extra_fields=(
                    "last_obs",
                    "truncation",
                    'episode_return'
                ),
            )
            return env_state, trajectory
        
        def sac_agent_rollout_fn(state:State):
            env_state, trajectory = rollout(
                env=self.env,
                agent=self.agent,
                env_state=state.env_state,
                agent_state=state.agent_state,
                key=rollout_key,
                rollout_length=self.config.rollout_length,
                env_extra_fields=(
                    "last_obs",
                    "truncation",
                    'episode_return'
                ),
            )
            return env_state, trajectory
        
        start_condition = jax.lax.lt(
            state.metrics.iterations, self.config.random_rollout_step
        )
        env_state, trajectory = jax.lax.cond(
            start_condition, random_agent_rollout_fn, sac_agent_rollout_fn, state
        )

        train_episode_return = average_episode_discount_return(
            trajectory.extras.env_extras.episode_return,
            trajectory.dones,
            pmap_axis_name=self.pmap_axis_name
        )
        trajectory = flatten_rollout_trajectory(trajectory)
        mask = trajectory.extras.env_extras.truncation.astype(bool)
        next_obs = jnp.where(
            mask[:, None],
            trajectory.extras.env_extras.last_obs,
            trajectory.next_obs,
        )
        trajectory = trajectory.replace(next_obs=next_obs)
        replay_buffer_state = self.replay_buffer.add(
            state.replay_buffer_state, trajectory
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
        state = state.update(
            env_state=env_state,
            replay_buffer_state=replay_buffer_state,
            agent_state=agent_state,
        )

        def real_update(state: State): 
            sampled_batch = self.replay_buffer.sample(state.replay_buffer_state, buffer_key)
            # def real_loss_fn(agent_state, sample_batch, key):
            #     loss_dict = self.agent.loss(agent_state, sample_batch, key)
            #     loss = loss_dict["actor_loss"] + loss_dict["critic_loss"]
            #     return loss, loss_dict
            
            # real_update_fn = agent_gradient_update(
            #     real_loss_fn,
            #     self.optimizer,
            #     pmap_axis_name=self.pmap_axis_name,
            #     has_aux=True
            # )
            # (loss, loss_dict), opt_state, agent_state = real_update_fn(
            #     state.opt_state,
            #     state.agent_state,
            #     sampled_batch.experience,
            #     learn_key
            # )
            def alpha_loss_fn(agent_state, sample_batch, key):
                loss_dict = self.agent.alpha_loss(agent_state, sample_batch, key)
                loss = loss_dict["alpha_loss"]
                return loss, loss_dict

            def critic_loss_fn(agent_state, sample_batch, key):
                loss_dict = self.agent.critic_loss(agent_state, sample_batch, key)
                loss = loss_dict["critic_loss"]
                return loss, loss_dict

            def actor_loss_fn(agent_state, sample_batch, key):
                loss_dict = self.agent.actor_loss(agent_state, sample_batch, key)
                loss = loss_dict["actor_loss"]
                return loss, loss_dict
            
            alpha_gradient_update_fn = alpha_agent_gradient_update(
                alpha_loss_fn,
                self.optimizer,
                pmap_axis_name=self.pmap_axis_name,
                has_aux=True,
            )

            critic_gradient_update_fn = critic_agent_gradient_update(
                critic_loss_fn,
                self.optimizer,
                pmap_axis_name=self.pmap_axis_name,
                has_aux=True,
            )

            actor_gradient_update_fn= actor_agent_gradient_update(
                actor_loss_fn,
                self.optimizer,
                pmap_axis_name=self.pmap_axis_name,
                has_aux=True,
            )

            agent_state = state.agent_state

            learn_key1, learn_key2, learn_key3 = jax.random.split(learn_key, num=3)

            alpha_loss_dict = PyTreeDict(alpha_loss=jnp.zeros(()), alpha=self.agent.alpha)
            alpha_opt_state = state.alpha_opt_state
            if self.agent.adaptive_alpha:
                (alpha_loss, alpha_loss_dict), alpha_opt_state, agent_state = (
                    alpha_gradient_update_fn(
                        state.alpha_opt_state,
                        agent_state,
                        sampled_batch.experience,
                        learn_key1,
                    )
                )

            (critic_loss, critic_loss_dict), critic_opt_state, agent_state = (
                critic_gradient_update_fn(
                    state.critic_opt_state,
                    agent_state,
                    sampled_batch.experience,
                    learn_key2
                )
            )

            (actor_loss, actor_loss_dict), actor_opt_state, agent_state = (
                actor_gradient_update_fn(
                    state.actor_opt_state,
                    agent_state,
                    sampled_batch.experience,
                    learn_key3
                )
            )

            target_critic_params = soft_target_update(
                agent_state.params.target_critic_params,
                agent_state.params.critic_params,
                self.config.tau
            )

            params = agent_state.params.replace(target_critic_params=target_critic_params)
            agent_state = agent_state.replace(params=params)
            state = state.update(
                agent_state=agent_state,
                critic_opt_state=critic_opt_state,
                actor_opt_state=actor_opt_state,
                alpha_opt_state=alpha_opt_state,
            )
            # state = state.update(
            #     agent_state=agent_state,
            #     opt_state=opt_state,
            # )

            train_metrics = TrainMetric(
                train_episode_return=train_episode_return,
                loss=critic_loss + actor_loss,
                raw_loss_dict=PyTreeDict(
                    actor_loss=actor_loss_dict["actor_loss"],
                    critic_loss=critic_loss_dict["critic_loss"],
                    entropy_loss=actor_loss_dict["entropy_loss"],
                    alpha_loss=alpha_loss_dict["alpha_loss"],
                    alpha=alpha_loss_dict["alpha"]
                )
            ).all_reduce(pmap_axis_name=self.pmap_axis_name)
            # train_metrics = TrainMetric(
            #     train_episode_return=train_episode_return,
            #     loss=loss,
            #     raw_loss_dict=loss_dict
            # ).all_reduce(pmap_axis_name=self.pmap_axis_name)

            return state, train_metrics
        
        def fake_update(state: State):
            train_metrics = TrainMetric(
                train_episode_return=train_episode_return,
                loss=jnp.zeros(()),
                raw_loss_dict=PyTreeDict(
                    actor_loss=jnp.zeros(()),
                    critic_loss=jnp.zeros(()),
                    entropy_loss=jnp.zeros(()),
                    alpha_loss=jnp.zeros(()),
                    alpha=jnp.zeros(())
                ),
            ).all_reduce(pmap_axis_name=self.pmap_axis_name)

            return state, train_metrics
        
        state, train_metrics = jax.lax.cond(
            self.replay_buffer.can_sample(replay_buffer_state),
            real_update,
            fake_update,
            state,
        )
        
        sampled_timesteps = psum(self.config.rollout_length * self.config.num_envs,
                                  axis_name=self.pmap_axis_name)

        workflow_metrics = WorkflowMetric(
            sampled_timesteps=state.metrics.sampled_timesteps+sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
        )


    def learn(self, state: State) -> State:
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        num_iters = math.ceil(self.config.total_timesteps / one_step_timesteps)
        start_iteration = tree_unpmap(state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, num_iters):
            train_metric, state = self.step(state)
            workflow_metric = state.metrics
            train_metric = tree_unpmap(train_metric, self.pmap_axis_name)
            train_metric_data = train_metric.to_local_dict()
            if train_metric.train_episode_return==MISSING_REWARD:
                del train_metric_data['train_episode_return']
            self.recorder.write(train_metric_data, i)
            workflow_metric = tree_unpmap(workflow_metric, self.pmap_axis_name)
            self.recorder.write(workflow_metric.to_local_dict(), i)

            if (i + 1) % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)
                eval_metrics = tree_unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write({"eval": eval_metrics.to_local_dict()}, i)
                logger.debug(eval_metrics)

            self.checkpoint_manager.save(
                i,
                args=ocp.args.StandardSave(
                    tree_unpmap(state, self.pmap_axis_name)),
            )

        return state
    
def loss_and_pgrad(
    loss_fn: Callable[..., float], pmap_axis_name: Optional[str], has_aux: bool = False
):
    g = jax.value_and_grad(loss_fn, has_aux=has_aux)

    def h(*args, **kwargs):
        value, grads = g(*args, **kwargs)
        return value, jax.lax.pmean(grads, axis_name=pmap_axis_name)

    return g if pmap_axis_name is None else h

def actor_agent_gradient_update(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    def _loss_fn(actor_params, agent_state, sample_batch, key):
        p = agent_state.params.replace(actor_params=actor_params)
        return loss_fn(agent_state.replace(params=p), sample_batch, key)

    loss_and_pgrad_fn = loss_and_pgrad(
        _loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
    )

    def f(opt_state, agent_state, *args, **kwargs):
        value, grads = loss_and_pgrad_fn(
            agent_state.params.actor_params, agent_state, *args, **kwargs
        )

        actor_params_update, opt_state = optimizer.update(grads, opt_state)
        updated_actor_params = optax.apply_updates(
            agent_state.params.actor_params, actor_params_update
        )
        updated_params = agent_state.params.replace(actor_params=updated_actor_params)
        agent_state = agent_state.replace(params=updated_params)

        return value, opt_state, agent_state

    return f


def critic_agent_gradient_update(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    def _loss_fn(critic_params, agent_state, sample_batch, key):
        p = agent_state.params.replace(critic_params=critic_params)
        return loss_fn(agent_state.replace(params=p), sample_batch, key)

    loss_and_pgrad_fn = loss_and_pgrad(
        _loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
    )

    def f(opt_state, agent_state, *args, **kwargs):
        value, grads = loss_and_pgrad_fn(
            agent_state.params.critic_params, agent_state, *args, **kwargs
        )

        critic_params_update, opt_state = optimizer.update(grads, opt_state)
        updated_critic_params = optax.apply_updates(
            agent_state.params.critic_params, critic_params_update
        )
        updated_params = agent_state.params.replace(critic_params=updated_critic_params)
        agent_state = agent_state.replace(params=updated_params)

        return value, opt_state, agent_state

    return f

def alpha_agent_gradient_update(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    def _loss_fn(alpha_params, agent_state, sample_batch, key):
        p = agent_state.params.replace(alpha_params=alpha_params)
        return loss_fn(agent_state.replace(params=p), sample_batch, key)

    loss_and_pgrad_fn = loss_and_pgrad(
        _loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
    )

    def f(opt_state, agent_state, *args, **kwargs):
        value, grads = loss_and_pgrad_fn(
            agent_state.params.alpha_params, agent_state, *args, **kwargs
        )

        alpha_params_update, opt_state = optimizer.update(grads, opt_state)
        updated_alpha_params = optax.apply_updates(
            agent_state.params.alpha_params, alpha_params_update
        )
        updated_params = agent_state.params.replace(alpha_params=updated_alpha_params)
        agent_state = agent_state.replace(params=updated_params)

        return value, opt_state, agent_state

    return f