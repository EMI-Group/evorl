import logging
import math
from typing import Any

import chex
import flashbax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from omegaconf import DictConfig

from evorl.distributed import psum, tree_pmean, tree_unpmap
from evorl.distributed.gradients import agent_gradient_update
from evorl.envs import AutoresetMode, Box, create_env, Space
from evorl.evaluator import Evaluator
from evorl.metrics import MetricBase, metricfield
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
from evorl.utils.jax_utils import scan_and_mean, tree_last, tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update
from evorl.workflows import OffPolicyRLWorkflow, skip_replay_buffer_state

from .agent import Agent, AgentState
from .random_agent import RandomAgent

logger = logging.getLogger(__name__)


class TD3TrainMetric(MetricBase):
    critic_loss: chex.Array
    actor_loss: chex.Array
    raw_loss_dict: LossDict = metricfield(
        default_factory=PyTreeDict, reduce_fn=tree_pmean
    )


class TD3NetworkParams(PyTreeData):
    """Contains training state for the learner."""

    actor_params: Params
    critic_params: Params
    target_actor_params: Params
    target_critic_params: Params


class TD3Agent(Agent):
    """
    The Agnet for DDPG
    """

    critic_hidden_layer_sizes: tuple[int] = (256, 256)
    actor_hidden_layer_sizes: tuple[int] = (256, 256)
    discount: float = 0.99
    exploration_epsilon: float = 0.5
    policy_noise: float = 0.2
    clip_policy_noise: float = 0.5

    normalize_obs: bool = False
    critic_network: nn.Module = pytree_field(lazy_init=True)  # nn.Module is ok
    actor_network: nn.Module = pytree_field(lazy_init=True)
    obs_preprocessor: Any = pytree_field(lazy_init=True, pytree_node=False)

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        obs_size = obs_space.shape[0]
        action_size = action_space.shape[0]

        key, critic_key, actor_key, obs_preprocessor_key = jax.random.split(key, num=4)

        # the output of the q_network is b*n_critics, n_critics is the number of critics, b is the batch size
        critic_network, critic_init_fn = make_q_network(
            obs_size=obs_size,
            action_size=action_size,
            n_stack=2,
            hidden_layer_sizes=self.critic_hidden_layer_sizes,
        )
        critic_params = critic_init_fn(critic_key)
        target_critic_params = critic_params

        # the output of the actor_network is b, b is the batch size
        actor_network, actor_init_fn = make_policy_network(
            action_size=action_size,
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes,
            activation_final=nn.tanh,
        )

        actor_params = actor_init_fn(actor_key)
        target_actor_params = actor_params

        self.set_frozen_attr("critic_network", critic_network)
        self.set_frozen_attr("actor_network", actor_network)

        params_state = TD3NetworkParams(
            critic_params=critic_params,
            actor_params=actor_params,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
        )

        # obs_preprocessor
        if self.normalize_obs:
            obs_preprocessor = running_statistics.normalize
            self.set_frozen_attr("obs_preprocessor", obs_preprocessor)
            dummy_obs = obs_space.sample(obs_preprocessor_key)
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state,
            action_space=action_space,
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        """
        Args:
            sample_barch: [#env, ...]
        used in sample action during rollout
        """
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        actions = self.actor_network.apply(agent_state.params.actor_params, obs)
        # add random noise
        noise = jax.random.normal(key, actions.shape) * self.exploration_epsilon
        actions += noise
        actions = jnp.clip(
            actions, agent_state.action_space.low, agent_state.action_space.high
        )

        return actions, PyTreeDict()

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        """
        Args:
            sample_barch: [#env, ...]
        """
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        actions = self.actor_network.apply(agent_state.params.actor_params, obs)

        return actions, PyTreeDict()

    def critic_loss(
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
        next_obs = sample_batch.extras.env_extras.last_obs
        obs = sample_batch.obs
        actions = sample_batch.actions

        if self.normalize_obs:
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        actions_next = self.actor_network.apply(
            agent_state.params.target_actor_params, next_obs
        )
        actions_next += jnp.clip(
            jax.random.normal(key, actions.shape) * self.policy_noise,
            -self.clip_policy_noise,
            self.clip_policy_noise,
        )

        # [B, 2]
        qs_next = self.critic_network.apply(
            agent_state.params.target_critic_params, next_obs, actions_next
        )
        qs_next_min = qs_next.min(-1)

        discounts = self.discount * (1 - sample_batch.extras.env_extras.termination)

        qs_target = sample_batch.rewards + discounts * qs_next_min
        qs_target = jnp.repeat(qs_target[..., None], 2, axis=-1)
        qs_target = jax.lax.stop_gradient(qs_target)

        qs = self.critic_network.apply(agent_state.params.critic_params, obs, actions)

        # q_loss = optax.huber_loss(qs, target_qs, delta=1).mean()
        q_loss = optax.squared_error(qs, qs_target).mean()

        return PyTreeDict(critic_loss=q_loss, q_value=qs.mean())

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
        obs = sample_batch.obs

        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # [T*B, A]
        actions = self.actor_network.apply(agent_state.params.actor_params, obs)

        # TODO: handle redundant computation
        # Note: what about using min_Q, like SAC?
        actor_loss = -jnp.mean(
            self.critic_network.apply(agent_state.params.critic_params, obs, actions)[
                ..., 0
            ]  # here, we use q1
        )
        return PyTreeDict(actor_loss=actor_loss)


class TD3Workflow(OffPolicyRLWorkflow):
    @classmethod
    def name(cls):
        return "TD3"

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
        """
        return workflow
        """
        env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
        )

        assert isinstance(
            env.action_space, Box
        ), "Only continue action space is supported."

        agent = TD3Agent(
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
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
        random_agent = RandomAgent(action_space=action_space, obs_space=obs_space)
        random_agent_state = random_agent.init(obs_space, action_space, key)

        # Note: in multi-devices mode, this method is running in pmap, and
        # config.num_envs = config.num_envs // num_devices
        # config.random_timesteps = config.random_timesteps // num_devices
        rollout_length = config.random_timesteps // config.num_envs
        env_state = self.env.reset(env_key)

        trajectory, env_state = rollout(
            env_fn=self.env.step,
            action_fn=random_agent.compute_actions,
            env_state=env_state,
            agent_state=random_agent_state,
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

        replay_buffer_state = self.replay_buffer.add(replay_buffer_state, trajectory)

        rollout_timesteps = rollout_length * config.num_envs
        sampled_timesteps = psum(
            jnp.uint32(rollout_timesteps), axis_name=self.pmap_axis_name
        )

        # ==== fill tansition state from init agent ====
        rollout_length = math.ceil(
            (config.learning_start_timesteps - rollout_timesteps) / config.num_envs
        )
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

        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    pmap_axis_name=self.pmap_axis_name,
                )
            )

        replay_buffer_state = self.replay_buffer.add(replay_buffer_state, trajectory)

        rollout_timesteps = rollout_length * config.num_envs
        sampled_timesteps += psum(
            jnp.uint32(rollout_timesteps), axis_name=self.pmap_axis_name
        )

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return state.replace(
            key=key, metrics=workflow_metrics, replay_buffer_state=replay_buffer_state
        )

    def step(self, state: State) -> tuple[TD3TrainMetric, State]:
        """
        the basic step function for the workflow to update agent
        """
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

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            critic_opt_state = opt_state.critic
            actor_opt_state = opt_state.actor

            key, critic_key, actor_key, rb_key = jax.random.split(key, num=4)

            if self.config.actor_update_interval - 1 > 0:

                def _sample_and_update_critic_fn(carry, unused_t):
                    key, agent_state, critic_opt_state = carry

                    key, rb_key, critic_key = jax.random.split(key, num=3)
                    # it's safe to use read-only replay_buffer_state here.
                    sampled_batch = self.replay_buffer.sample(
                        replay_buffer_state, rb_key
                    ).experience

                    (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
                        critic_update_fn(
                            critic_opt_state, agent_state, sampled_batch, critic_key
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

            sampled_batch = self.replay_buffer.sample(
                replay_buffer_state, rb_key
            ).experience

            (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
                critic_update_fn(
                    critic_opt_state, agent_state, sampled_batch, critic_key
                )
            )

            (actor_loss, actor_loss_dict), agent_state, actor_opt_state = (
                actor_update_fn(actor_opt_state, agent_state, sampled_batch, actor_key)
            )

            target_actor_params = soft_target_update(
                agent_state.params.target_actor_params,
                agent_state.params.actor_params,
                self.config.tau,
            )
            target_critic_params = soft_target_update(
                agent_state.params.target_critic_params,
                agent_state.params.critic_params,
                self.config.tau,
            )
            agent_state = agent_state.replace(
                params=agent_state.params.replace(
                    target_actor_params=target_actor_params,
                    target_critic_params=target_critic_params,
                )
            )

            opt_state = opt_state.replace(
                actor=actor_opt_state, critic=critic_opt_state
            )

            return (
                (key, agent_state, opt_state),
                (critic_loss, actor_loss, critic_loss_dict, actor_loss_dict),
            )

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

        train_metrics = TD3TrainMetric(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        # calculate the numbner of timestep
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

    def _multi_steps(self, state):
        def _step(state, _):
            train_metrics, state = self.step(state)
            return state, train_metrics

        state, train_metrics = jax.lax.scan(
            _step, state, (), length=self.config.fold_iters
        )
        train_metrics = tree_last(train_metrics)
        return train_metrics, state

    def learn(self, state: State) -> State:
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        sampled_timesteps = tree_unpmap(state.metrics.sampled_timesteps).tolist()
        num_iters = math.ceil(
            (self.config.total_timesteps - sampled_timesteps)
            / (one_step_timesteps * self.config.fold_iters)
        )

        for i in range(num_iters):
            train_metrics, state = self._multi_steps(state)
            workflow_metrics = state.metrics

            # current iteration
            iterations = tree_unpmap(
                state.metrics.iterations, self.pmap_axis_name
            ).tolist()
            train_metrics = tree_unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = tree_unpmap(workflow_metrics, self.pmap_axis_name)
            self.recorder.write(train_metrics.to_local_dict(), iterations)
            self.recorder.write(workflow_metrics.to_local_dict(), iterations)

            if iterations % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)
                eval_metrics = tree_unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write({"eval": eval_metrics.to_local_dict()}, iterations)

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
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )
        cls._multi_steps = jax.jit(cls._multi_steps, static_argnums=(0,))


def clean_trajectory(trajectory: SampleBatch):
    """
    clean the trajectory to make it suitable for the replay buffer
    """
    return trajectory.replace(
        next_obs=None,
        dones=None,
    )
