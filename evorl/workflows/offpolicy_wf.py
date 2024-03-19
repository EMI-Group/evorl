import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig
from evox import Stateful, State

from evorl.envs import Env
from evorl.agents import Agent
from evorl.types import RolloutMetric, SampleBatch

import flashbax

"""
    Single Agent Off-Policy Reinforcement Learning Workflow
"""


class OffPolicyRLWorkflow(Stateful):
    def __init__(
        self,
        config: DictConfig,
        agent: Agent,
        env: Env,
    ):
        self.config = config
        self.agent = agent
        self.env = env  # batched env
        self.optimizer = optax.adam(config.optimizer.lr)
        self.replay_buffer = flashbax.make_flat_buffer(
            max_length=config.replay_buffer.capacity,
            min_length=config.replay_buffer.min_size,
            sample_batch_size=config.train_batch_size,
            add_batch_size=config.num_envs*config.rollout_length
        )

    def setup(self, key):
        key, agent_key, env_key, buffer_key = jax.random.split(key, 3)
        agent_state = self.agent.init(agent_key)

        env_state = self.env.reset(env_key)

        replay_buffer_state = init_replay_buffer(
            self.env, env_state, self.replay_buffer, buffer_key)

        return State(
            key=key,
            rollout_metric=RolloutMetric(),
            replay_buffer_state=replay_buffer_state,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=self.optimizer.init(agent_state.params)
        )

    def step(self, state):
        return state.update()


def init_replay_buffer(env, env_state, replay_buffer, key):
    num_envs = jax.tree_leaves(env_state)[0].shape[0]
    # dummy_action = jnp.tile(env.action_space.sample(), 
    dummy_action = env.action_space.sample()
    dummy_action = jnp.broadcast_to(dummy_action, (num_envs, *dummy_action.shape))
    dummy_obs = env.obs_space.sample()
    dummy_obs = jnp.broadcast_to(dummy_action, (num_envs, *dummy_action.shape))

    # TODO: handle RewardDict
    dummy_reward = jnp.zeros((num_envs))
    dummy_done = jnp.zeros((num_envs))

    dummy_nest_obs = dummy_obs

    # Customize your algorithm's stored data
    dummy_sample_batch = SampleBatch(
        obs=dummy_obs,
        action=dummy_action,
        reward=dummy_reward,
        next_obs=dummy_nest_obs,
        done=dummy_done
    )

    replay_buffer_state = replay_buffer.init(dummy_sample_batch)
    return replay_buffer_state
