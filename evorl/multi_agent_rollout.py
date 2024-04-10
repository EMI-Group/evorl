import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map
import chex
from flax import struct
from typing import (
    Any, Union, Tuple, Optional, Sequence, Mapping
)
from evorl.agents import Agent, AgentState
from evorl.types import (
    Reward, RewardDict, ExtraInfo, PyTreeData, PyTreeDict, AgentID
)
from evorl.sample_batch import SampleBatch, Episode
from evorl.envs import Env, EnvState
from evorl.utils.ma_utils import batchify, unbatchify, multi_agent_episode_done
from functools import partial

# TODO: add RNN Policy support

# Decentralized Execution
def decentralized_env_step(
    env: Env,
    agents: Mapping[AgentID, Agent],
    env_state: EnvState,
    agent_states: Mapping[AgentID, AgentState],  # readonly
    sample_batch: SampleBatch,
    key: chex.PRNGKey,
    env_extra_fields: Sequence[str] = (),
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect one-step data.
    """

    num_agents = len(agents)
    env_keys = jax.random.split(key, num_agents)

    actions = {}
    policy_extras = {}

    # assume agents have different models, non-parallel
    for (agent_id, agent), env_key in zip(agents.items(), env_keys):
        agent_sample_batch = SampleBatch(
            obs = sample_batch.obs[agent_id]
        )
        actions[agent_id], policy_extras[agent_id] = agent.compute_actions(
            agent_states[agent_id], agent_sample_batch, env_key)
        

    env_nstate = env.step(env_state, actions)

    info = env_nstate.info
    env_extras = {x: info[x] for x in env_extra_fields if x in info}

    transition = SampleBatch(
        obs=env_state.obs,
        actions=actions,
        rewards=env_nstate.reward,
        dones=env_nstate.done,
        next_obs=env_nstate.obs,
        extras=PyTreeDict(
            policy_extras=policy_extras,
            env_extras=env_extras
        )
    )

    return env_nstate, transition


def decentralized_rollout(
    env: Env,
    agents: Mapping[AgentID, Agent],
    env_state: EnvState,
    agent_states: Mapping[AgentID, AgentState],  # readonly
    key: chex.PRNGKey,
    rollout_length: int,
    env_extra_fields: Sequence[str] = ()
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect given rollout_length trajectory.
        Tips: when use jax.jit, use: jax.jit(partial(rollout, env, agent))

        Args:
            env: vmapped env w/ autoreset

        Returns:
            env_state: last env_state after rollout
            trajectory: SampleBatch [T, B, ...], T=rollout_length, B=#envs
    """

    def _one_step_rollout(carry, unused_t):
        """
            sample_batch: one-step obs
            transition: one-step full info
        """
        env_state, current_key = carry
        next_key, current_key = jax.random.split(current_key, 2)

        # sample_batch: [#envs, ...]
        sample_batch = SampleBatch(
            obs=env_state.obs
        )

        # transition: [#envs, ...]
        env_nstate, transition = decentralized_env_step(
            env, agents,
            env_state, agent_states,
            sample_batch, current_key, env_extra_fields
        )

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key), (),
        length=rollout_length
    )

    return env_state, trajectory


def centralized_env_step(
    env: Env,
    agent: Mapping[AgentID, Agent],
    env_state: EnvState,
    agent_state: Mapping[AgentID, AgentState],  # readonly
    sample_batch: SampleBatch,
    key: chex.PRNGKey,
    env_extra_fields: Sequence[str] = (),
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect one-step data.
    """

    num_agents = len(agents)
    env_keys = jax.random.split(key, num_agents)

    actions = {}
    policy_extras = {}

    # assume agents have different models, non-parallel
    for (agent_id, agent), env_key in zip(agents.items(), env_keys):
        agent_sample_batch = SampleBatch(
            obs = sample_batch.obs[agent_id]
        )
        actions[agent_id], policy_extras[agent_id] = agent.compute_actions(
            agent_states[agent_id], agent_sample_batch, env_key)
        

    env_nstate = env.step(env_state, actions)

    info = env_nstate.info
    env_extras = {x: info[x] for x in env_extra_fields if x in info}

    transition = SampleBatch(
        obs=env_state.obs,
        actions=actions,
        rewards=env_nstate.reward,
        dones=env_nstate.done,
        next_obs=env_nstate.obs,
        extras=PyTreeDict(
            policy_extras=policy_extras,
            env_extras=env_extras
        )
    )

    return env_nstate, transition


def centralized_rollout(
    env: Env,
    agents: Mapping[AgentID, Agent],
    env_state: EnvState,
    agent_states: Mapping[AgentID, AgentState],  # readonly
    key: chex.PRNGKey,
    rollout_length: int,
    padding: bool = False,
    env_extra_fields: Sequence[str] = ()
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect given rollout_length trajectory.
        Tips: when use jax.jit, use: jax.jit(partial(rollout, env, agent))

        Args:
            env: vmapped env w/ autoreset

        Returns:
            env_state: last env_state after rollout
            trajectory: SampleBatch [T, B, ...], T=rollout_length, B=#envs
    """

    def _one_step_rollout(carry, unused_t):
        """
            sample_batch: one-step obs
            transition: one-step full info
        """
        env_state, current_key = carry
        next_key, current_key = jax.random.split(current_key, 2)

        # sample_batch: [#envs, ...]
        sample_batch = SampleBatch(
            obs=env_state.obs
        )

        # transition: [#envs, ...]
        env_nstate, transition = decentralized_env_step(
            env, agents,
            env_state, agent_states,
            sample_batch, current_key, env_extra_fields
        )

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key), (),
        length=rollout_length
    )

    return env_state, trajectory