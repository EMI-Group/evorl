import jax
import jax.numpy as jnp
from flax import struct


from .agent import Agent, AgentState
from evorl.networks import make_q_network
from evorl.workflows import OffPolicyRLWorkflow
from evorl.envs import create_env, Discrete
from evorl.types import TrainMetric
from evorl.evaluator import Evaluator

from evox import State

from omegaconf import DictConfig
from typing import Dict, Tuple, Sequence
import optax
import chex
import distrax
import dataclasses

import flashbax

from evorl.types import (
    EnvLike, LossDict, Action, Params, PolicyExtraInfo, EnvState,
    Observation, SampleBatch
)

import logging

logger = logging.getLogger(__name__)


@struct.dataclass
class A2CNetworkParams:
    """Contains training state for the learner."""
    q_params: Params


@dataclasses.dataclass
class DQNAgent(Agent):
    """
        Double-DQN
    """
    q_hidden_layer_sizes: Tuple[int] = (256, 256)
    discount: float = 0.99
    eploration_epsilon: float = 0.1

    def init(self, key: chex.PRNGKey) -> AgentState:
        obs_size = self.obs_space.shape[0]
        action_size = self.action_space.n

        self.q_network, q_init_fn = make_q_network(
            obs_size=obs_size,
            action_size=action_size,
            hidden_layer_sizes=self.q_hidden_layer_sizes,
            n_critics=1
        )

        key, q_key = jax.random.split(key)

        q_params = q_init_fn(q_key)

        return AgentState(
            params=A2CNetworkParams(q_params)
        )

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        """
            Args:
                sample_barch: [#env, ...]
        """

        qs = self.q_network.apply(
            agent_state.params.q_params, sample_batch.obs)

        # TODO: use tfp.Distribution
        actions_dist = distrax.EpsilonGreedy(
            qs, epsilon=self.eploration_epsilon)
        actions = actions_dist.sample(seed=key)

        return actions, dict(
            q_values=qs
        )

    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        """
            Args:
                sample_barch: [#env, ...]
        """
        qs = self.q_network.apply(
            agent_state.params.q_params, sample_batch.obs)

        actions_dist = distrax.EpsilonGreedy(
            qs, epsilon=self.eploration_epsilon)
        actions = actions_dist.mode()

        return actions, {}

    def loss(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
        """
            Args:
                sample_barch: [B, ...]
        """
        # TODO: impl it
        qs = self.q_network.apply(
            agent_state.params.q_params, sample_batch.obs)

        td_error = None

        return dict(
            q_loss=td_error)


class DQNWorkflow(OffPolicyRLWorkflow):
    @staticmethod
    def _rescale_config(config, devices) -> None:
        num_devices = len(devices)

        #TODO: impl it

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env,
            config.env_type,
            episode_length=1000,
            parallel=config.num_envs,
            autoreset=True
        )
        

        assert isinstance(env.action_space, Discrete), "Only Discrete action space is supported."

        agent = DQNAgent(
            action_space=env.action_space,
            obs_space=env.obs_space,
            q_hidden_layer_sizes=config.agent_network.q_hidden_layer_sizes,
            discount=config.discount,
            eploration_epsilon=config.eploration_epsilon
        )

        optimizer = optax.adam(config.optimizer.lr)

        replay_buffer = flashbax.make_flat_buffer(
            max_length=config.replay_buffer.capacity,
            min_length=config.replay_buffer.min_size,
            sample_batch_size=config.train_batch_size,
            add_batch_size=config.num_envs*config.rollout_length
        )

        batch_shape = (config.num_envs*config.rollout_length,)
        def _replay_buffer_init_fn(replay_buffer, key):
            # dummy_action = jnp.tile(env.action_space.sample(),
            dummy_action = env.action_space.sample()
            dummy_action = jnp.broadcast_to(
                dummy_action, batch_shape+dummy_action.shape)
            
            action_shape = jax.eval_shape(env.action_space.sample)
            dummy_action = jnp.zeros(batch_shape+action_shape)

            obs_shape = jax.eval_shape(env.obs_space.sample)
            dummy_obs = jnp.zeors(batch_shape+obs_shape)

            # TODO: handle RewardDict
            dummy_reward = jnp.zeros(batch_shape)
            dummy_done = jnp.zeros(batch_shape)

            dummy_nest_obs = dummy_obs

            # Customize your algorithm's stored data
            dummy_sample_batch = SampleBatch(
                obs=dummy_obs,
                actions=dummy_action,
                rewards=dummy_reward,
                next_obs=dummy_nest_obs,
                dones=dummy_done
            )

            replay_buffer_state = replay_buffer.init(dummy_sample_batch)

            return replay_buffer_state
        

        eval_env = create_env(
            config.env,
            config.env_type,
            episode_length=1000,
            parallel=config.num_eval_envs,
            autoreset=False
        )

        evaluator = Evaluator(
            env=eval_env, agent=agent, max_episode_length=1000)

        return cls(env, agent, optimizer, evaluator, replay_buffer, _replay_buffer_init_fn, config)




