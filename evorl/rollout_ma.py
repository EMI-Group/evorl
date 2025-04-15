from collections.abc import Callable, Mapping, Sequence
from functools import partial

import chex
import jax

from evorl.agent import Agent, AgentState
from evorl.envs import EnvState, MultiAgentEnv
from evorl.sample_batch import SampleBatch
from evorl.types import AgentID, PyTreeDict
from evorl.utils.jax_utils import tree_get
from evorl.utils.ma_utils import batchify, unbatchify

# TODO: add RNN Policy support

# Decentralized Execution

__all__ = [
    "decentralized_env_step",
    "decentralized_rollout",
    "decentralized_env_step_with_shared_model",
    "decentralized_rollout_with_shared_model",
]


def decentralized_env_step(
    env: MultiAgentEnv,
    agents: Mapping[AgentID, Agent],
    env_state: EnvState,
    agent_states: Mapping[AgentID, AgentState],  # readonly
    sample_batch: SampleBatch,
    key: chex.PRNGKey,
    env_extra_fields: Sequence[str] = (),
) -> tuple[EnvState, SampleBatch]:
    """Collect one-step data."""
    num_agents = len(agents)
    env_keys = jax.random.split(key, num_agents)

    actions = {}
    policy_extras = {}

    # sample_batch: [#envs, ...]
    sample_batch = SampleBatch(obs=env_state.obs)

    # assume agents have different models, non-parallel
    for (agent_id, agent), env_key in zip(agents.items(), env_keys):
        agent_sample_batch = SampleBatch(obs=tree_get(sample_batch.obs, agent_id))
        actions[agent_id], policy_extras[agent_id] = agent.compute_actions(
            agent_states[agent_id], agent_sample_batch, env_key
        )

    env_nstate = env.step(env_state, actions)

    info = env_nstate.info
    env_extras = {x: info[x] for x in env_extra_fields if x in info}

    transition = SampleBatch(
        obs=env_state.obs,
        actions=actions,
        rewards=env_nstate.reward,
        dones=env_nstate.done,
        next_obs=env_nstate.obs,
        extras=PyTreeDict(policy_extras=policy_extras, env_extras=env_extras),
    )

    return env_nstate, transition


def decentralized_rollout(
    env: MultiAgentEnv,
    agents: Mapping[AgentID, Agent],
    env_state: EnvState,
    agent_states: Mapping[AgentID, AgentState],  # readonly
    key: chex.PRNGKey,
    rollout_length: int,
    env_extra_fields: Sequence[str] = (),
) -> tuple[EnvState, SampleBatch]:
    """Collect given rollout_length trajectory.

    Tips: when use jax.jit, use: jax.jit(partial(rollout, env, agent))

    Args:
        env: vmapped env w/ autoreset

    Returns:
        env_state: last env_state after rollout
        trajectory: SampleBatch [T, B, ...], T=rollout_length, B=#envs
    """

    def _one_step_rollout(carry, unused_t):
        env_state, current_key = carry
        next_key, current_key = jax.random.split(current_key, 2)

        # transition: [#envs, ...]
        env_nstate, transition = decentralized_env_step(
            env,
            agents,
            env_state,
            agent_states,
            current_key,
            env_extra_fields,
        )

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key), (), length=rollout_length
    )

    return env_state, trajectory


def decentralized_env_step_with_shared_model(
    env: MultiAgentEnv,
    agent: Agent,
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    key: chex.PRNGKey,
    obs_batchify_fn: Callable[[jax.Array], Mapping[AgentID, jax.Array]],
    action_unbatchify_fn: Callable[[jax.Array], Mapping[AgentID, jax.Array]],
    env_extra_fields: Sequence[str] = (),
) -> tuple[EnvState, SampleBatch]:
    """Collect one-step data."""
    # sample_batch: [#envs, ...]
    sample_batch = SampleBatch(obs=obs_batchify_fn(env_state.obs))

    batched_actions, policy_extras = agent.compute_actions(
        agent_state, sample_batch, key
    )
    actions = action_unbatchify_fn(batched_actions)

    env_nstate = env.step(env_state, actions)

    # policy_extras = unbatchify(policy_extras, env.agents)

    env_nstate = env.step(env_state, actions)

    info = env_nstate.info
    env_extras = {x: info[x] for x in env_extra_fields if x in info}

    transition = SampleBatch(
        obs=env_state.obs,
        actions=actions,
        rewards=env_nstate.reward,
        dones=env_nstate.done,
        next_obs=env_nstate.obs,
        extras=PyTreeDict(policy_extras=policy_extras, env_extras=env_extras),
    )

    return env_nstate, transition


def decentralized_rollout_with_shared_model(
    env: MultiAgentEnv,
    agent: Agent,
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    key: chex.PRNGKey,
    rollout_length: int,
    obs_batchify_fn: None | (Callable[[jax.Array], Mapping[AgentID, jax.Array]]) = None,
    action_unbatchify_fn: None
    | (Callable[[jax.Array], Mapping[AgentID, jax.Array]]) = None,
    env_extra_fields: Sequence[str] = (),
) -> tuple[EnvState, SampleBatch]:
    """Centrialized Execution: Collect given rollout_length trajectory.

    Args:
        env: vmapped env w/ autoreset

    Returns:
        env_state: last env_state after rollout
        trajectory: SampleBatch [T, B, ...], T=rollout_length, B=#envs
    """
    if obs_batchify_fn is None:
        obs_batchify_fn = partial(batchify, agent_list=env.agents)

    if action_unbatchify_fn is None:
        action_unbatchify_fn = partial(unbatchify, agent_list=env.agents)

    def _one_step_rollout(carry, unused_t):
        env_state, current_key = carry
        next_key, current_key = jax.random.split(current_key, 2)

        # transition: [#envs, ...]
        env_nstate, transition = decentralized_env_step_with_shared_model(
            env,
            agent,
            env_state,
            agent_state,
            current_key,
            obs_batchify_fn,
            action_unbatchify_fn,
            env_extra_fields,
        )

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key), (), length=rollout_length
    )

    return env_state, trajectory
