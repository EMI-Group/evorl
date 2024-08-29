import copy
import logging
import math
from collections.abc import Sequence
from typing_extensions import Self  # pytype: disable=not-supported-yet

import chex
import flashbax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import flax.linen as nn
import optax


from omegaconf import DictConfig, read_write as read_write_cfg

from evorl.distributed import psum, agent_gradient_update
from evorl.metrics import MetricBase
from evorl.types import (
    PyTreeData,
    PyTreeDict,
    PyTreeNode,
    State,
    Params,
    pytree_field,
    Action,
    PolicyExtraInfo,
    LossDict,
)
from evorl.utils import running_statistics
from evorl.utils.jax_utils import scan_and_mean, tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update
from evorl.evaluator import Evaluator
from evorl.sample_batch import SampleBatch
from evorl.agent import Agent, AgentState, RandomAgent
from evorl.networks import make_q_network, MLP, ActivationFn
from evorl.envs import Space, create_env, AutoresetMode, Box
from evorl.rollout import rollout
from evorl.workflows import OffPolicyWorkflow

from ..offpolicy_utils import clean_trajectory
from ..td3 import TD3Agent


logger = logging.getLogger(__name__)


class TrainMetric(MetricBase):
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array
    pop_train_metrics: MetricBase


class EvalMetric(MetricBase):
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array


class PopTD3NetworkParams(PyTreeData):
    """Contains training state for the learner."""

    pop_actor_params: Params
    critic_params: Params
    target_pop_actor_params: Params
    target_critic_params: Params


def make_policy_network(
    action_size: int,
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu,
    activation_final: ActivationFn | None = None,
) -> nn.Module:
    """Creates a batched policy network."""
    policy_model = nn.vmap(
        MLP,
        variable_axes={"params": 0},
        split_rngs={"params": True},
    )(
        layer_sizes=tuple(hidden_layer_sizes) + (action_size,),
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        activation_final=activation_final,
    )

    def init_fn(rng):
        return policy_model.init(rng, jnp.zeros((1, obs_size)))

    return policy_model, init_fn


def flatten_rollout_pop_trajectory(trajectory: SampleBatch):
    """
    Flatten the trajectory from [#pop, T, B, ...] to [#pop*T*B, ...]
    """
    return jtu.tree_map(lambda x: jax.lax.collapse(x, 0, 3), trajectory)


class PopTD3Agent(TD3Agent):
    agent_state_pytree_axes: chex.ArrayTree = pytree_field(
        lazy_init=True, pytree_node=False
    )

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        obs_size = obs_space.shape[0]
        action_size = action_space.shape[0]

        key, critic_key, actor_key, obs_preprocessor_key = jax.random.split(key, num=4)

        # global critic network
        # the output of the q_network is (b, n_critics), n_critics is the number of critics, b is the batch size
        critic_network, critic_init_fn = make_q_network(
            obs_size=obs_size,
            action_size=action_size,
            n_stack=2,
            hidden_layer_sizes=self.critic_hidden_layer_sizes,
        )
        critic_params = critic_init_fn(critic_key)
        target_critic_params = critic_params

        # pop actor networks
        # the output of the actor_network is (b,), b is the batch size
        actor_network, actor_init_fn = make_policy_network(
            action_size=action_size,
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes,
            activation_final=nn.tanh,
        )

        pop_actor_params = actor_init_fn(actor_key)
        target_pop_actor_params = pop_actor_params

        self.set_frozen_attr("critic_network", critic_network)
        self.set_frozen_attr("actor_network", actor_network)

        params_state = PopTD3NetworkParams(
            critic_params=critic_params,
            pop_actor_params=pop_actor_params,
            target_critic_params=target_critic_params,
            target_pop_actor_params=target_pop_actor_params,
        )

        # shared obs_preprocessor
        if self.normalize_obs:
            obs_preprocessor = running_statistics.normalize
            self.set_frozen_attr("obs_preprocessor", obs_preprocessor)
            dummy_obs = obs_space.sample(obs_preprocessor_key)
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
        else:
            obs_preprocessor_state = None

        self.set_frozen_attr(
            "agent_state_pytree_axes",
            AgentState(
                params=PopTD3NetworkParams(
                    pop_actor_params=0,
                    target_pop_actor_params=0,
                    critic_params=None,
                    target_critic_params=None,
                ),
                obs_preprocessor_state=None,
                action_space=None,
            ),
        )

        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state,
            action_space=action_space,
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        # obs: (#pop, B, ...)
        pop_size = sample_batch.obs.shape[0]
        act_keys = jax.random.split(key, pop_size)

        return jax.vmap(
            super().compute_actions, in_axes=(self.agent_state_pytree_axes, 0, 0)
        )(agent_state, sample_batch, act_keys)

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        # obs: (#pop, B, ...)
        pop_size = sample_batch.obs.shape[0]
        act_keys = jax.random.split(key, pop_size)

        return jax.vmap(
            super().evaluate_actions, in_axes=(self.agent_state_pytree_axes, 0, 0)
        )(agent_state, sample_batch, act_keys)

    def actor_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        pop_size = sample_batch.obs.shape[0]
        loss_keys = jax.random.split(key, pop_size)

        return jax.vmap(
            super().actor_loss, in_axes=(self.agent_state_pytree_axes, None, 0)
        )(agent_state, sample_batch, loss_keys)


class CEMRLWorkflow(OffPolicyWorkflow):
    """
    1 critic + n actors + 1 replay buffer.
    We use shard_map to split and parallel the population.
    """

    @classmethod
    def name(cls):
        return "CEM-RL"

    @classmethod
    def enable_jit(cls) -> None:
        super().enable_jit()
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )

    @classmethod
    def enable_pmap(cls, axis_name) -> None:
        """
        Use shard_map instead
        """
        pass

    @property
    def enable_multi_devices(self) -> bool:
        return self.sharding is not None

    @staticmethod
    def _rescale_config(config) -> None:
        num_devices = jax.device_count()

        if config.pop_size % num_devices != 0:
            logger.warning(
                f"pop_size({config.pop_size}) cannot be divided by num_devices({num_devices}), "
                f"rescale pop_size to {config.pop_size // num_devices * num_devices}"
            )

        config.pop_size = (config.pop_size // num_devices) * num_devices

    @classmethod
    def build_from_config(
        cls,
        config: DictConfig,
        enable_multi_devices: bool = True,
        enable_jit: bool = True,
    ) -> Self:
        config = copy.deepcopy(config)  # avoid in-place modification

        # devices = jax.local_devices()

        # always enable multi-devices

        with read_write_cfg(config):
            cls._rescale_config(config)

        workflow = cls._build_from_config(config)

        # mesh = Mesh(devices, axis_names=(POP_AXIS_NAME,))

        # workflow.devices = devices
        # workflow.sharding = NamedSharding(mesh, P(POP_AXIS_NAME))

        return workflow

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        """
        return workflow
        """

        # env for one actor
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

        agent = PopTD3Agent(
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

        # to evaluate the pop-mean actor
        eval_env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        eval_agent = TD3Agent(
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
        )

        evaluator = Evaluator(
            env=eval_env,
            agent=eval_agent,
            max_episode_steps=config.env.max_episode_steps,
        )

        return cls(
            env,
            agent,
            optimizer,
            evaluator,
            replay_buffer,
            config,
        )

    def setup(self, key: chex.PRNGKey) -> State:
        key, agent_key, env_key, rb_key = jax.random.split(key, 4)

        # [#pop, ...]
        agent_state, opt_state = self._setup_agent_and_optimizer(agent_key)

        # agent_state = jax.device_put(agent_state, self.sharding)
        # opt_state = jax.device_put(opt_state, self.sharding)

        workflow_metrics = self._setup_workflow_metrics()

        env_key = jax.random.split(env_key, self.config.pop_size)
        env_state = jax.vmap(self.env.reset)(env_key)

        replay_buffer_state = self._setup_replaybuffer(rb_key)

        state = State(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state,
            replay_buffer_state=replay_buffer_state,
        )

        logger.info("Start replay buffer post-setup")

        state = self._postsetup_replaybuffer(state)

        logger.info("Complete replay buffer post-setup")

        return state

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

        def _fill_pop(agent, agent_state, key, rollout_length):
            env_key, rollout_key = jax.random.split(key)
            env_state = self.env.reset(env_key)

            if self.config.fitness_with_exploration:
                action_fn = agent.compute_actions
            else:
                action_fn = agent.evaluate_actions

            trajectory, env_state = rollout(
                env_fn=jax.vmap(self.env.step),
                action_fn=action_fn,
                env_state=env_state,
                agent_state=agent_state,
                key=rollout_key,
                rollout_length=rollout_length,
                env_extra_fields=("last_obs", "termination"),
            )

            # [T, #pop, B, ...] -> [#pop*T*B, ...]
            trajectory = clean_trajectory(trajectory)
            trajectory = flatten_rollout_pop_trajectory(trajectory)
            trajectory = tree_stop_gradient(trajectory)

            return trajectory

        def _fill(agent, agent_state, key, rollout_length):
            env_key, rollout_key = jax.random.split(key)
            env_state = self.env.reset(env_key)

            # Note: not needed for random agent
            if self.config.fitness_with_exploration:
                action_fn = agent.compute_actions
            else:
                action_fn = agent.evaluate_actions

            trajectory, env_state = rollout(
                env_fn=self.env.step,
                action_fn=action_fn,
                env_state=env_state,
                agent_state=agent_state,
                key=rollout_key,
                rollout_length=rollout_length,
                env_extra_fields=("last_obs", "termination"),
            )

            # [T, B, ...] -> [T*B, ...]
            trajectory = clean_trajectory(trajectory)
            trajectory = flatten_rollout_trajectory(trajectory)
            trajectory = tree_stop_gradient(trajectory)

            return trajectory

        def _update_obs_preprocessor(agent_state, trajectory):
            if (
                agent_state.obs_preprocessor_state is not None
                and len(trajectory.obs) > 0
            ):
                agent_state = agent_state.replace(
                    obs_preprocessor_state=running_statistics.update(
                        agent_state.obs_preprocessor_state,
                        trajectory.obs,
                        pmap_axis_name=self.pmap_axis_name,
                    )
                )
            return agent_state

        # ==== fill random transitions ====

        key, random_rollout_key, rollout_key = jax.random.split(state.key, num=3)
        random_agent = RandomAgent()
        random_agent_state = random_agent.init(
            obs_space, action_space, jax.random.PRNGKey(0)
        )
        rollout_length = config.random_timesteps // config.num_envs

        trajectory = _fill(
            random_agent,
            random_agent_state,
            key=random_rollout_key,
            rollout_length=rollout_length,
        )

        replay_buffer_state = self.replay_buffer.add(replay_buffer_state, trajectory)
        agent_state = _update_obs_preprocessor(agent_state, trajectory)

        rollout_timesteps = rollout_length * config.num_envs
        sampled_timesteps = psum(
            jnp.uint32(rollout_timesteps), axis_name=self.pmap_axis_name
        )

        # ==== fill tansition state from init agent ====
        rollout_length = math.ceil(
            (config.learning_start_timesteps - rollout_timesteps)
            / (config.pop_size * config.num_envs)
        )

        trajectory = _fill_pop(
            self.agent,
            agent_state,
            key=rollout_key,
            rollout_length=rollout_length,
        )

        replay_buffer_state = self.replay_buffer.add(replay_buffer_state, trajectory)
        agent_state = _update_obs_preprocessor(agent_state, trajectory)

        rollout_timesteps = rollout_length * config.num_envs
        sampled_timesteps = sampled_timesteps + psum(
            jnp.uint32(rollout_timesteps), axis_name=self.pmap_axis_name
        )

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
        )

    def step(self, state: State) -> tuple[TrainMetric, State]:
        """
        the basic step function for the workflow to update agent
        """
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        if self.config.fitness_with_exploration:
            action_fn = self.agent.compute_actions
        else:
            action_fn = self.agent.evaluate_actions

        # the trajectory [T, #pop, B, ...]
        trajectory, env_state = rollout(
            env_fn=jax.vmap(self.env.step),
            action_fn=action_fn,
            env_state=state.env_state,
            agent_state=state.agent_state,
            key=rollout_key,
            rollout_length=self.config.rollout_length,
            env_extra_fields=("last_obs", "termination"),
        )

        trajectory = clean_trajectory(trajectory)
        trajectory = flatten_rollout_pop_trajectory(trajectory)
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
            # sample_batch: (B, ...)
            loss_dict = self.agent.critic_loss(agent_state, sample_batch, key)

            loss = self.config.loss_weights.critic_loss * loss_dict.critic_loss
            return loss, loss_dict

        def actor_loss_fn(agent_state, sample_batch, key):
            # different actor shares same sample_batch (B, ...) input
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

        train_metrics = TrainMetric(
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


class CEMState(PyTreeData):
    mean: chex.ArrayTree
    variance: chex.ArrayTree


class CEM(PyTreeNode):
    agent: Agent

    def init(key: chex.PRNGKey) -> CEMState:
        pass

    def update(state: CEMState, fitness: chex.Array) -> CEMState:
        pass

    def sample(state: CEMState, key: chex.PRNGKey) -> chex.ArrayTree:
        pass
