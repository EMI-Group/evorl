from chex._src.pytypes import PRNGKey
import jax
import jax.numpy as jnp
from flax import struct
from flax.training import checkpoints
import orbax.checkpoint as ocp
import math

from .agent import Agent, AgentState
from evorl.workflows import OffPolicyRLWorkflow
from evorl.agents.random_agent import RandomAgent
from evorl.envs import create_env, Box, Env, EnvState
from evorl.sample_batch import SampleBatch
from evorl.evaluator import Evaluator
from evorl.utils import running_statistics
from evorl.rollout import rollout, env_step
from evox import State
from evorl.networks import MLP
from evorl.distributed import PMAP_AXIS_NAME, split_key_to_devices, tree_unpmap
from evorl.utils.toolkits import average_episode_discount_return, soft_target_update
from omegaconf import DictConfig, OmegaConf
from typing import Tuple, Any, Sequence, Callable, Optional
from evorl.utils.orbax_utils import load
from evorl.distributed.gradients import agent_gradient_update
import optax
import chex
import distrax
import dataclasses
import os
from evorl.metrics import TrainMetric, WorkflowMetric
import flashbax
from evorl.types import (
    LossDict,
    Action,
    Params,
    PolicyExtraInfo,
    PyTreeDict,
    pytree_field,
)
import logging
import flax.linen as nn
from tensorboardX import SummaryWriter
from jax.profiler import trace, TraceAnnotation

logger = logging.getLogger(__name__)
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@struct.dataclass
class TD3NetworkParams:
    """Contains training state for the learner."""

    params: Params
    target_params: Params


class TD3Agent(Agent):
    """
    TD3
    """

    critic_hidden_layer_sizes: Tuple[int] = (256, 256)
    actor_hidden_layer_sizes: Tuple[int] = (256, 256)
    discount: float = 1
    exploration_epsilon: float = 0.5
    normalize_obs: bool = False
    critic_network: nn.Module = pytree_field(lazy_init=True)  # nn.Module is ok
    actor_network: nn.Module = pytree_field(lazy_init=True)
    obs_preprocessor: Any = pytree_field(lazy_init=True, pytree_node=False)
    n_critics: int = 2
    policy_noise: float = 0.2
    policy_noise_clip: float = 0.5

    def init(self, key: chex.PRNGKey) -> AgentState:
        obs_size = self.obs_space.shape[0]
        action_size = self.action_space.shape[0]

        # the output of the q_network is b*n_critics, n_critics is the number of critics, b is the batch size
        critic_network, critic_init_fn = make_critic_networks(
            obs_size=obs_size,
            action_size=action_size,
            hidden_layer_sizes=self.critic_hidden_layer_sizes,
            n_critics=self.n_critics,
        )

        key, q_key, actor_key, obs_preprocessor_key = jax.random.split(key, num=4)
        q_keys = jax.random.split(q_key, self.n_critics)
        critic_params = critic_init_fn(q_keys)
        target_critic_params = critic_params

        # the output of the actor_network is b, b is the batch size
        action_scale = jnp.array((self.action_space.high - self.action_space.low) / 2.0)
        action_bias = jnp.array((self.action_space.high + self.action_space.low) / 2.0)
        actor_network, actor_init_fn = make_actor_network(
            action_size=action_size,
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes,
            action_scale=action_scale,
            action_bias=action_bias,
        )

        actor_params = actor_init_fn(actor_key)
        target_actor_params = actor_params

        self.set_frozen_attr("critic_network", critic_network)
        self.set_frozen_attr("actor_network", actor_network)

        params_state = TD3NetworkParams(
            params=PyTreeDict(critic_params=critic_params, actor_params=actor_params),
            target_params=PyTreeDict(
                critic_params=target_critic_params, actor_params=target_actor_params
            ),
        )
        # obs_preprocessor
        if self.normalize_obs:
            self.obs_preprocessor = running_statistics.normalize
            dummy_obs = self.obs_space.sample(obs_preprocessor_key)
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state, obs_preprocessor_state=obs_preprocessor_state
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> Tuple[Action, PolicyExtraInfo]:
        """
        Args:
            sample_barch: [#env, ...]
        used in sample action during rollout
        """
        action = self.actor_network.apply(
            agent_state.params.params.actor_params, sample_batch.obs
        )
        # add random noise
        noise_stddev = (
            self.action_space.high - self.action_space.low
        ) * self.exploration_epsilon
        noise = jax.random.normal(key, action.shape) * noise_stddev
        action += noise
        action = jnp.clip(action, self.action_space.low, self.action_space.high)

        return jax.lax.stop_gradient(action), PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> Tuple[Action, PolicyExtraInfo]:
        """
        Args:
            sample_barch: [#env, ...]
        """
        action = self.actor_network.apply(
            agent_state.params.target_params.actor_params, sample_batch.obs
        )

        return jax.lax.stop_gradient(action), PyTreeDict()

    def cirtic_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        """
        Args:
            sample_barch: [B, ...]

        Return: LossDict[
            actor_loss
            critic_loss
            actor_entropy_loss
        ]
        """
        next_obs = sample_batch.next_obs
        obs = sample_batch.obs
        actions = sample_batch.actions

        if self.normalize_obs:
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        next_actions = self.actor_network.apply(
            agent_state.params.target_params.actor_params, next_obs
        )

        # random noise    self.action_space next_actions.shape
        noise = jnp.clip(
            jax.random.normal(key, next_actions.shape) * self.policy_noise,
            -self.policy_noise_clip,
            self.policy_noise_clip,
        ) * (self.action_space.high - self.action_space.low)

        next_actions = jnp.clip(
            next_actions + noise, self.action_space.low, self.action_space.high
        )

        next_data = jnp.concatenate([next_obs, next_actions], axis=-1)
        q_net_batch_apply = jax.vmap(
            lambda x: self.critic_network.apply(x, next_data), in_axes=0
        )
        next_qs = q_net_batch_apply(agent_state.params.target_params.critic_params).squeeze(-1)

        min_next_q = jnp.min(next_qs, axis=0)
        target_qs = (
            sample_batch.rewards + self.discount * (1 - sample_batch.dones) * min_next_q
        )
        target_qs = jax.lax.stop_gradient(target_qs)

        input_data = jnp.concatenate([obs, actions], axis=-1)
        q_net_batch_apply = jax.vmap(
            lambda x: self.critic_network.apply(x, input_data), in_axes=0
        )
        qs = q_net_batch_apply(agent_state.params.params.critic_params).squeeze(-1)

        # another way to compute the lose
        # q_loss = optax.huber_loss(qs, target_qs, delta=1).mean()
        q_loss = ((qs - target_qs) ** 2).mean(axis=-1).sum()

        return PyTreeDict(critic_loss=q_loss)

    def actor_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        """
        Args:
            sample_barch: [B, ...]

        Return: LossDict[
            actor_loss
            critic_loss
            actor_entropy_loss
        ]
        """
        next_obs = sample_batch.next_obs
        obs = sample_batch.obs

        if self.normalize_obs:
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # [T*B, A]
        gen_actions = self.actor_network.apply(agent_state.params.params.actor_params, obs)

        q0_params = jax.tree_util.tree_map(
            lambda x: x[0], agent_state.params.params.critic_params
        )
        actor_loss = -jnp.mean(
            self.critic_network.apply(
                q0_params, jnp.concatenate([obs, gen_actions], axis=-1)
            ).squeeze(-1)
        )
        return PyTreeDict(actor_loss=actor_loss)

    def loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        """
        Args:
            sample_barch: [B, ...]

        Return: LossDict[
            actor_loss
            critic_loss
            actor_entropy_loss
        ]
        """

        q_loss = self.cirtic_loss(agent_state, sample_batch, key).critic_loss

        # ====== actor =======

        # [T*B, A]
        actor_loss = self.actor_loss(agent_state, sample_batch, key).actor_loss

        return PyTreeDict(
            actor_loss=actor_loss,
            critic_loss=q_loss,
        )

    def compute_values(
        self, agent_state: AgentState, sample_batch: SampleBatch
    ) -> chex.Array:
        """
        Args:
            obs: [B, ...]

        Return: [B, ...]
        """
        obs = sample_batch.obs
        actions = sample_batch.actions
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        input_data = jnp.concatenate([obs, actions], axis=-1)
        return jnp.min(
            self.critic_network.apply(agent_state.params.params.critic_params, input_data), axis=0
        )


class TD3Workflow(OffPolicyRLWorkflow):

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
        """
        return workflow
        """
        env = create_env(
            env_name=config.env.env_name,
            env_type=config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset=True,
            discount=config.discount,
        )

        assert isinstance(
            env.action_space, Box
        ), "Only continue action space is supported."

        agent = TD3Agent(
            action_space=env.action_space,
            obs_space=env.obs_space,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            n_critics=config.agent_network.n_critics,
            policy_noise=config.agent_network.policy_noise,
            policy_noise_clip=config.agent_network.policy_noise_clip,
        )

        # one optimizer, two opt_states (in setup function) for both actor and critic
        optimizer = optax.adam(learning_rate=config.optimizer.lr)

        replay_buffer = flashbax.make_item_buffer(
            max_length=config.replay_buffer.capacity // jax.device_count(),
            min_length=config.replay_buffer.min_size,
            sample_batch_size=config.replay_buffer.sample_batch_size,
            add_batches=True,
        )

        def _replay_buffer_init_fn(key):
            # create dummy data to initialize the replay buffer
            dummy_action = jnp.zeros(env.action_space.shape)
            dummy_obs = jnp.zeros(env.obs_space.shape)

            dummy_reward = jnp.zeros(())
            dummy_done = jnp.zeros(())
            dummy_nest_obs = dummy_obs
            dummy_last_obs = dummy_obs
            dummy_sample_batch = SampleBatch(
                obs=dummy_obs,
                actions=dummy_action,
                rewards=dummy_reward,
                next_obs=dummy_nest_obs,
                dones=dummy_done,
                extras=PyTreeDict(
                    policy_extras=PyTreeDict(),
                    env_extras=PyTreeDict(
                        {"last_obs": dummy_last_obs, "truncation": dummy_done}
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
            autoreset=True,
            discount=config.discount,
        )

        evaluator = Evaluator(env=eval_env, agent=agent, max_episode_steps=1000)

        return cls(
            env,
            agent,
            optimizer,
            evaluator,
            replay_buffer,
            _replay_buffer_init_fn,
            config,
        )

    @classmethod
    def enable_jit(cls, device: jax.Device = None) -> None:
        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls.fill_replay_buffer = jax.jit(cls.fill_replay_buffer, static_argnums=(0,))
        cls._step = jax.jit(cls._step, static_argnums=(0,))
        # return super().enable_jit(device)

    def fill_replay_buffer(self, state: State) -> State:
        key, rollout_key, agent_key = jax.random.split(state.key, num=3)
        random_agent = RandomAgent(
            action_space=self.env.action_space, obs_space=self.env.obs_space
        )
        random_state = random_agent.init(agent_key)

        env_state, trajectory = rollout(
            env=self.env,
            agent=random_agent,
            env_state=state.env_state,
            agent_state=random_state,
            key=rollout_key,
            rollout_length=self.config.learning_starts,
            env_extra_fields=(
                "last_obs",
                "truncation",
            ),
        )
        trajectory = jax.tree_util.tree_map(
            lambda x: jax.lax.collapse(x, 0, 2), trajectory
        )

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

        state = state.update(
            env_state=env_state, replay_buffer_state=replay_buffer_state, key=key
        )
        # get episode return
        train_episode_return = env_state.info.episode_return.mean()
        loss = jnp.zeros(())
        loss_dict = dict(
            actor_loss=loss,
            critic_loss=loss,
        )
        train_metrics = TrainMetric(
            train_episode_return=train_episode_return,
            loss=loss,
            raw_loss_dict=loss_dict,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        # calculate the numbner of timestep
        sampled_timesteps = (
            state.metrics.sampled_timesteps
            + self.config.rollout_length * self.config.num_envs
        )

        # iterations is the number of updates of the agent
        workflow_metrics = WorkflowMetric(
            sampled_timesteps=sampled_timesteps,
            iterations=state.metrics.iterations + self.config.num_envs,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.update(metrics=workflow_metrics)

    def setup(self, key: chex.PRNGKey) -> State:
        self.recorder.init()

        key, agent_key, env_key, buffer_key = jax.random.split(key, 4)

        agent_state = self.agent.init(agent_key)

        workflow_metrics = self._setup_workflow_metrics()

        opt_state = self.optimizer.init(agent_state.params)

        if self.enable_multi_devices:
            (
                workflow_metrics,
                agent_state,
                opt_state,
                replay_buffer_state,
            ) = jax.device_put_replicated(
                (
                    workflow_metrics,
                    agent_state,
                    opt_state,
                    replay_buffer_state,
                ),
                self.devices,
            )

            # key and env_state should be different over devices
            key = split_key_to_devices(key, self.devices)

            env_key = split_key_to_devices(env_key, self.devices)
            env_state = jax.pmap(self.env.reset, axis_name=self.pmap_axis_name)(env_key)
            buffer_key = split_key_to_devices(buffer_key, self.devices)
            replay_buffer_state = jax.pmap(
                self._init_replay_buffer, axis_name=self.pmap_axis_name
            )(buffer_key)
        else:
            env_state = self.env.reset(env_key)
            replay_buffer_state = self._init_replay_buffer(buffer_key)

        return State(
            key=key,
            metrics=workflow_metrics,
            replay_buffer_state=replay_buffer_state,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state,
        )

    def _step(self, state: State) -> Tuple[TrainMetric, State]:
        """
        the basic step function for the workflow to update agent
        """
        key, rollout_key, learn_key, buffer_key = jax.random.split(state.key, num=4)

        # the trajectory (T*B*variable dim), dim T = 1 (default) and B = num_envs
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
            ),
        )
        trajectory = jax.tree_util.tree_map(
            lambda x: jax.lax.collapse(x, 0, 2), trajectory
        )
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
        sampled_batch = self.replay_buffer.sample(replay_buffer_state, buffer_key)

        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    pmap_axis_name=self.pmap_axis_name,
                )
            )

        def critic_loss_fn(agent_state, sample_batch, key):
            loss_dict = PyTreeDict(
                actor_loss=jnp.zeros(()),
                critic_loss=self.agent.cirtic_loss(agent_state, sample_batch, key)["critic_loss"],
            )
            loss = loss_dict["critic_loss"]
            return loss, loss_dict

        def both_loss_fn(agent_state, sample_batch, key):
            actor_key, critic_key = jax.random.split(key)
            actor_loss_dict = self.agent.actor_loss(
                agent_state, sample_batch, actor_key
            )
            critic_loss_dict = self.agent.cirtic_loss(
                agent_state, sample_batch, critic_key
            )
            loss_dict = PyTreeDict(
                actor_loss=actor_loss_dict["actor_loss"],
                critic_loss=critic_loss_dict["critic_loss"],
            )
            loss = loss_dict["actor_loss"] + loss_dict["critic_loss"]
            return loss, loss_dict

        critic_gradient_update = agent_gradient_update(
            critic_loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True,
        )

        both_gradient_update = agent_gradient_update(
            both_loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True,
        )

        def update_critic(agent_state):
            (loss, loss_dict), opt_state, agent_state = (
                critic_gradient_update(
                    state.opt_state,
                    agent_state,
                    sampled_batch.experience,
                    learn_key,
                )
            )
            return (
                loss,
                loss_dict,
                opt_state,
                agent_state,
            )

        def update_both(agent_state):
            (loss, loss_dict), opt_state, agent_state = (
                both_gradient_update(
                    state.opt_state,
                    agent_state,
                    sampled_batch.experience,
                    learn_key,
                )
            )

            target_params = soft_target_update(
                agent_state.params.target_params,
                agent_state.params.params,
                self.config.tau,
            )

            params = agent_state.params.replace(
                target_params=target_params,
            )
            agent_state = agent_state.replace(params=params)

            return (
                loss,
                loss_dict,
                opt_state,
                agent_state,
            )

        update_condition = (
            state.metrics.iterations % self.config.actor_update_interval
        ) == 0
       
        loss, loss_dict, opt_state, agent_state = (
            jax.lax.cond(update_condition, update_both, update_critic, agent_state)
        )

        state = state.update(
            env_state=env_state,
            replay_buffer_state=replay_buffer_state,
            agent_state=agent_state,
            opt_state=opt_state,
        )

        # get episode return, in DDPG the return is the rewards
        train_episode_return = env_state.info.episode_return.mean()

        train_metrics = TrainMetric(
            train_episode_return=train_episode_return,
            loss=loss,
            raw_loss_dict=loss_dict,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        # calculate the numbner of timestep
        sampled_timesteps = (
            state.metrics.sampled_timesteps
            + self.config.rollout_length * self.config.num_envs
        )

        # iterations is the number of updates of the agent
        workflow_metrics = WorkflowMetric(
            sampled_timesteps=sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.update(
            key=key,
            metrics=workflow_metrics,
        )

    def step(self, state: State) -> Tuple[TrainMetric, State]:
        if state.metrics.iterations < self.config.learning_starts:
            train_metrics, state = self.fill_replay_buffer(state)
        else:
            train_metrics, state = self._step(state)

        return train_metrics, state

    def learn(self, state: State) -> State:
        # one_step_timesteps = self.config.rollout_length * self.config.num_envs
        num_iters = self.config.total_timesteps
        start_iteration = tree_unpmap(state.metrics.iterations, self.pmap_axis_name)
        # jax.profiler.start_server(9999)
        # jax.profiler.start_trace("/tensorboard", create_perfetto_link=True)
        ckpt_path = self.config.load_path + "/checkpoints"
        if self.config.check_load and os.path.isdir(ckpt_path):
            ckpt_options = ocp.CheckpointManagerOptions(
                save_interval_steps=self.config.checkpoint.save_interval_steps,
                max_to_keep=self.config.checkpoint.max_to_keep,
            )
            logger.info(f"Set loadiong checkpoint path: {ckpt_path}")
            checkpoint_manager = ocp.CheckpointManager(
                ckpt_path,
                options=ckpt_options,
                metadata=OmegaConf.to_container(self.config),  # Rescaled real config
            )
            last_step = checkpoint_manager.latest_step()
            state = load(path=ckpt_path, state=state)
            start_iteration = tree_unpmap(last_step, self.pmap_axis_name)

        with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
            for i in range(start_iteration, num_iters):
                train_metrics, state = self.step(state)
                workflow_metrics = state.metrics

                # the log_interval should be odd due to the frequency of updating actor is even
                if (i + 1) % self.config.log_interval == 0:
                    train_metrics = tree_unpmap(train_metrics, self.pmap_axis_name)
                    self.recorder.write(train_metrics.to_local_dict(), i)
                    workflow_metrics = tree_unpmap(
                        workflow_metrics, self.pmap_axis_name
                    )
                    self.recorder.write(workflow_metrics.to_local_dict(), i)

                if (i + 1) % self.config.eval_interval == 0:
                    eval_metrics, state = self.evaluate(state)
                    eval_metrics = tree_unpmap(eval_metrics, self.pmap_axis_name)
                    self.recorder.write({"eval": eval_metrics.to_local_dict()}, i)

                self.checkpoint_manager.save(
                    i,
                    args=ocp.args.StandardSave(tree_unpmap(state, self.pmap_axis_name)),
                )       

        logger.info("finish!")
        return state

    def load(self, state: State):
        ckpt_options = ocp.CheckpointManagerOptions(
            save_interval_steps=self.config.checkpoint.save_interval_steps,
            max_to_keep=self.config.checkpoint.max_to_keep,
        )
        ckpt_path = self.config.load_path + "/checkpoints"
        logger.info(f"Set loadiong checkpoint path: {ckpt_path}")
        checkpoint_manager = ocp.CheckpointManager(
            ckpt_path,
            options=ckpt_options,
            metadata=OmegaConf.to_container(self.config),  # Rescaled real config
        )
        last_step = checkpoint_manager.latest_step()
        reload_state = checkpoint_manager.restore(
            last_step,
            args=ocp.args.StandardRestore(tree_unpmap(state, self.pmap_axis_name)),
        )
        logger.info(f"Reloaded from step {last_step}")

class Actor(nn.Module):
    """
    the network for actor
    """

    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    action_scale: jnp.ndarray = 1.0
    action_bias: jnp.ndarray = 0.0

    @nn.compact
    def __call__(self, data):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
        hidden = nn.tanh(hidden)
        hidden = hidden * self.action_scale + self.action_bias
        return hidden


def make_actor_network(
    action_size: int,
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    action_scale: jnp.ndarray = 1.0,
    action_bias: jnp.ndarray = 0.0,
):

    actor = Actor(
        layer_sizes=list(hidden_layer_sizes) + [action_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        action_scale=action_scale,
        action_bias=action_bias,
    )
    init_fn = lambda rng: actor.init(rng, jnp.ones((1, obs_size)))

    return actor, init_fn


def make_critic_networks(
    obs_size: int,
    action_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    n_critics: int = 2,  # abandon
) -> nn.Module:
    network = MLP(
        layer_sizes=list(hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    init_fn = jax.vmap(
        lambda x: network.init(x, jnp.ones((1, obs_size + action_size))), in_axes=(0,)
    )

    return network, init_fn
