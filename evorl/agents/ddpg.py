from chex._src.pytypes import PRNGKey
import jax
import jax.numpy as jnp
from flax import struct
import math

from .agent import Agent, AgentState
from evorl.networks import make_q_network, make_policy_network
from evorl.workflows import OffPolicyRLWorkflow
from evorl.envs import create_env, Box, Env, EnvState
from evorl.sample_batch import SampleBatch
from evorl.evaluator import Evaluator
from evorl.utils import running_statistics
from evorl.rollout import rollout, env_step
from evox import State
from evorl.networks import MLP
from evorl.distributed import agent_gradient_update, pmean
from evorl.utils.toolkits import average_episode_discount_return, soft_target_update
from omegaconf import DictConfig
from typing import Tuple, Any, Sequence, Callable
import optax
import chex
import distrax
import dataclasses
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

logger = logging.getLogger(__name__)
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]


@struct.dataclass
class DDPGNetworkParams:
    """Contains training state for the learner."""

    q_params: Params
    target_q_params: Params
    actor_params: Params
    target_actor_params: Params


class DDPGAgent(Agent):
    """
    DDPG
    """

    q_hidden_layer_sizes: Tuple[int] = (256, 256)
    actor_hidden_layer_sizes: Tuple[int] = (256, 256)
    discount: float = 1
    exploration_epsilon: float = 0.1
    normalize_obs: bool = False
    q_network: nn.Module = pytree_field(lazy_init=True)  # nn.Module is ok
    actor_network: nn.Module = pytree_field(lazy_init=True)
    obs_preprocessor: Any = pytree_field(lazy_init=True, pytree_node=False)

    def init(self, key: chex.PRNGKey) -> AgentState:
        obs_size = self.obs_space.shape[0]
        action_size = self.action_space.shape[0]

        # the output of the q_network is b*n_critics, n_critics is the number of critics, b is the batch size
        q_network, q_init_fn = make_q_network(
            obs_size=obs_size,
            action_size=action_size,
            hidden_layer_sizes=self.q_hidden_layer_sizes,
            n_critics=1,
        )

        key, q_key, actor_key, obs_preprocessor_key = jax.random.split(key, num=4)

        q_params = q_init_fn(q_key)
        target_q_params = q_params

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

        self.set_frozen_attr("q_network", q_network)
        self.set_frozen_attr("actor_network", actor_network)

        params_state = DDPGNetworkParams(
            q_params=q_params,
            target_q_params=target_q_params,
            actor_params=actor_params,
            target_actor_params=target_actor_params,
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
            agent_state.params.actor_params, sample_batch.obs
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
            agent_state.params.target_actor_params, sample_batch.obs
        )

        return jax.lax.stop_gradient(action), PyTreeDict()

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

        next_obs = sample_batch.next_obs
        obs = sample_batch.obs
        actions = sample_batch.actions

        if self.normalize_obs:
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # ======= critic =======
        actor = self.actor_network.apply(agent_state.params.target_actor_params, obs)
       

        next_qs = self.q_network.apply(
            agent_state.params.target_q_params, next_obs, actor
        ).squeeze(-1)

        target_qs = (
            sample_batch.rewards + self.discount * (1 - sample_batch.dones) * next_qs
        )
        target_qs = jax.lax.stop_gradient(target_qs)
        qs = self.q_network.apply(agent_state.params.q_params, obs, actions).squeeze(-1)

        # in DDPG, we use the target network to compute the target value and in cleanrl, the lose is MSE loss
        # q_loss = optax.huber_loss(qs, target_qs, delta=1).mean()
        q_loss = ((qs - target_qs) ** 2).mean()

        # ====== actor =======

        # [T*B, A]
        gen_actions = self.actor_network.apply(agent_state.params.actor_params, obs)

        actor_loss = -jnp.mean(
            self.q_network.apply(agent_state.params.q_params, obs, gen_actions).squeeze(
                -1
            )
        )

        return dict(
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

        return self.q_network.apply(
            agent_state.params.value_params, obs, actions
        ).squeeze(-1)

    def compute_random_actions(
        self, key: chex.PRNGKey, sample_batch: SampleBatch
    ) -> Tuple[Action, PolicyExtraInfo]:
        """
        get random actions for exploration used before "learning_starts" condition is met.
        """
        action = jax.random.uniform(
            key,
            shape=(sample_batch.obs.shape[0],) + self.action_space.shape,
            minval=self.action_space.low,
            maxval=self.action_space.high,
        )
        policy_extras = PyTreeDict()
        return jax.lax.stop_gradient(action), policy_extras


class DDPGWorkflow(OffPolicyRLWorkflow):

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

        agent = DDPGAgent(
            action_space=env.action_space,
            obs_space=env.obs_space,
            q_hidden_layer_sizes=config.agent_network.q_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
        )

        optimizer = optax.adam(learning_rate=config.optimizer.lr)

        replay_buffer = flashbax.make_item_buffer(
            max_length=config.replay_buffer.capacity,
            min_length=config.num_envs * config.rollout_length,
            sample_batch_size=config.num_envs * config.rollout_length,
            add_batches=True,
        )

        def _replay_buffer_init_fn(replay_buffer, key):
            # create dummy data to initialize the replay buffer
            dummy_action = jnp.zeros(env.action_space.shape)
            dummy_obs = jnp.zeros(env.obs_space.shape)

            dummy_reward = jnp.zeros(())
            dummy_done = jnp.zeros(())
            dummy_nest_obs = dummy_obs

            dummy_sample_batch = SampleBatch(
                obs=dummy_obs,
                actions=dummy_action,
                rewards=dummy_reward,
                next_obs=dummy_nest_obs,
                dones=dummy_done,
                extras=PyTreeDict(policy_extras=PyTreeDict(), env_extras=PyTreeDict()),
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

    def step(self, state: State) -> Tuple[TrainMetric, State]:
        """
        Args:
            state: State
        """
        key, rollout_key, learn_key, buffer_key = jax.random.split(state.key, num=4)

        def fill_replay_buffer(state: State):
            replay_buffer_state = state.replay_buffer_state
            env_state = state.env_state
            sample_batch = SampleBatch(obs=env_state.obs)
            actions, _ = self.agent.compute_random_actions(rollout_key, sample_batch)
            env_nstate = self.env.step(env_state, actions)
            trajectory = SampleBatch(
                obs=env_state.obs,
                actions=actions,
                rewards=env_nstate.reward,
                next_obs=env_nstate.obs,
                dones=env_nstate.done,
                extras=PyTreeDict(policy_extras=PyTreeDict(), env_extras=PyTreeDict()),
            )
            # transition = jax.tree_util.tree_map(lambda x: jax.lax.collapse(x,0,2), transition)
            replay_buffer_state = self.replay_buffer.add(
                replay_buffer_state, trajectory
            )
            env_state = env_nstate
            state.update(env_state=env_state, replay_buffer_state=replay_buffer_state)

            # get episode return, in DDPG the return is the rewards
            train_episode_return = average_episode_discount_return(
                env_state.info.episode_return, trajectory.dones
            ).mean()

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

            return state, train_metrics

        def normal_step(state: State):
            # the trajectory (T*B*variable dim), dim T = 1 (default) and B = num_envs
            env_state, trajectory = rollout(
                env=self.env,
                agent=self.agent,
                env_state=state.env_state,
                agent_state=state.agent_state,
                key=rollout_key,
                rollout_length=self.config.rollout_length,
                env_extra_fields=("episode_return"),
            )
            trajectory = jax.tree_util.tree_map(
                lambda x: jax.lax.collapse(x, 0, 2), trajectory
            )
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

            condition = (
                state.metrics.iterations % self.config.actor_update_interval
            ) == 0

            @jax.jit
            def loss_fn(agent_state, sample_batch, key):
                loss_dict = self.agent.loss(agent_state, sample_batch, key)
                loss_weights = self.config.optimizer.loss_weights

                def get_both_loss():
                    return (
                        loss_weights["actor_loss"] * loss_dict["actor_loss"]
                        + loss_weights["critic_loss"] * loss_dict["critic_loss"]
                    )

                def get_critic_loss():
                    return loss_weights["critic_loss"] * loss_dict["critic_loss"]

                # condition = (state.metrics.iterations % self.config.actor_update_interval) == 0
                loss = jax.lax.cond(condition, get_both_loss, get_critic_loss)
                return loss, loss_dict

            update_fn = agent_gradient_update(
                loss_fn,
                self.optimizer,
                pmap_axis_name=self.pmap_axis_name,
                has_aux=True,
            )

            (loss, loss_dict), opt_state, agent_state = update_fn(
                state.opt_state, agent_state, sampled_batch.experience, learn_key
            )

            def update_target_params():
                target_q_params = soft_target_update(
                    agent_state.params.target_q_params,
                    agent_state.params.q_params,
                    self.config.tau,
                )
                target_actor_params = soft_target_update(
                    agent_state.params.target_actor_params,
                    agent_state.params.actor_params,
                    self.config.tau,
                )
                params = agent_state.params.replace(
                    target_q_params=target_q_params,
                    target_actor_params=target_actor_params,
                )
                return params

            def keep_target_params():
                return agent_state.params

            params = jax.lax.cond(condition, update_target_params, keep_target_params)

            agent_state = agent_state.replace(params=params)

            state = state.update(
                env_state=env_state,
                replay_buffer_state=replay_buffer_state,
                agent_state=agent_state,
                opt_state=opt_state,
            )

            # get episode return, in DDPG the return is the rewards
            train_episode_return = average_episode_discount_return(
                env_state.info.episode_return, trajectory.dones
            ).mean()

            train_metrics = TrainMetric(
                train_episode_return=train_episode_return,
                loss=loss,
                raw_loss_dict=loss_dict,
            ).all_reduce(pmap_axis_name=self.pmap_axis_name)

            return state, train_metrics

        learn_start = jnp.floor(
            self.config.learning_starts
            / self.config.num_envs
            / self.config.rollout_length
        ).astype(state.metrics.iterations.dtype)
        condition = jax.lax.lt(state.metrics.iterations, learn_start)
        state, train_metrics = jax.lax.cond(
            condition, fill_replay_buffer, normal_step, state
        )

        # calculate the numbner of timestep
        sampled_timesteps = (
            state.metrics.sampled_timesteps
            + self.config.rollout_length * self.config.num_envs
        )

        # workflow_metrics
        workflow_metrics = WorkflowMetric(
            sampled_timesteps=sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.update(key=key, metrics=workflow_metrics,  sample_actions = self.agent.compute_actions(state.agent_state, SampleBatch(obs=state.env_state.obs), key)[0],)

    def learn(self, state: State) -> State:
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        num_iters = math.ceil(self.config.total_timesteps / one_step_timesteps)
        for i in range(num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            if (i + 1) % self.config.log_interval == 0:
                logger.info(workflow_metrics)
                logger.info(train_metrics)
                # logger.info(state.sample_actions.flatten())
                logger.info("action value max: {}".format(jnp.amax(state.sample_actions,axis=0)))
                logger.info("action value min: {}".format(jnp.amin(state.sample_actions,axis=0)))

            if (i + 1) % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)
                logger.info(eval_metrics)

        return state

    # def fill_replay_buffer(self, state: State) -> State:
    #     key = state.key
    #     key, loop_key = jax.random.split(key, num=2)
    #     env_state = state.env_state
    #     replay_buffer_state = state.replay_buffer_state
    #     sample_batch = SampleBatch(
    #         obs=env_state.obs
    #     )

    #     def step_fn(carry, _):
    #         env_state, replay_buffer_state, loop_key = carry
    #         loop_key, current_key = jax.random.split(loop_key)
    #         actions, _ = self.agent.compute_random_actions(current_key, sample_batch)
    #         env_nstate = self.env.step(env_state, actions)
    #         transition = SampleBatch(
    #             obs=env_state.obs,
    #             actions=actions,
    #             rewards=env_nstate.reward,
    #             next_obs=env_nstate.obs,
    #             dones=env_nstate.done,
    #             extras=PyTreeDict(
    #                 policy_extras=PyTreeDict(),
    #                 env_extras=PyTreeDict()
    #             )
    #         )
    #         # transition = jax.tree_util.tree_map(lambda x: jax.lax.collapse(x,0,2), transition)
    #         replay_buffer_state = self.replay_buffer.add(replay_buffer_state, transition)
    #         return (env_nstate, replay_buffer_state, loop_key), None

    #     (env_state, replay_buffer_state, loop_key), _ = jax.lax.scan(
    #         step_fn, (env_state, replay_buffer_state, loop_key), (),length=self.config.learning_starts
    #     )
    #     state = state.update(env_state=env_state, key=key, replay_buffer_state=replay_buffer_state)

    #     return state


class Actor(MLP):
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
