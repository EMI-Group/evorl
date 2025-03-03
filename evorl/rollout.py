from collections.abc import Sequence
from functools import partial
from typing import Protocol

import chex
import jax
import jax.numpy as jnp

from evorl.agent import AgentActionFn, AgentState
from evorl.envs import EnvState, EnvStepFn
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.jax_utils import rng_split

# TODO: add RNN Policy support

__all__ = [
    "rollout",
    "eval_rollout_episode",
    "fast_eval_rollout_episode",
]


class RolloutFn(Protocol):
    def __call__(
        self,
        env_fn: EnvStepFn,
        action_fn: AgentActionFn,
        env_state: EnvState,
        agent_state: AgentState,
        key: chex.PRNGKey,
        rollout_length: int,
        *args,
        **kwargs,
    ):
        pass


def env_step(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    key: chex.PRNGKey,
    env_extra_fields: Sequence[str] = (),
) -> tuple[SampleBatch, EnvState]:
    """Collect one-step data."""
    # sample_batch: [#envs, ...]
    sample_batch = SampleBatch(obs=env_state.obs)

    actions, policy_extras = action_fn(agent_state, sample_batch, key)
    env_nstate = env_fn(env_state, actions)

    info = env_nstate.info
    env_extras = PyTreeDict({x: info[x] for x in env_extra_fields if x in info})

    transition = SampleBatch(
        obs=env_state.obs,
        actions=actions,
        rewards=env_nstate.reward,
        dones=env_nstate.done,
        next_obs=env_nstate.obs,
        extras=PyTreeDict(policy_extras=policy_extras, env_extras=env_extras),
    )

    return transition, env_nstate


def eval_env_step(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    key: chex.PRNGKey,
) -> tuple[SampleBatch, EnvState]:
    """Collect one-step data in evaluation mode."""
    # sample_batch: [#envs, ...]
    sample_batch = SampleBatch(obs=env_state.obs)

    actions, policy_extras = action_fn(agent_state, sample_batch, key)
    env_nstate = env_fn(env_state, actions)

    transition = SampleBatch(
        rewards=env_nstate.reward,
        dones=env_nstate.done,
    )

    return transition, env_nstate


def rollout(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    env_extra_fields: Sequence[str] = (),
) -> tuple[SampleBatch, EnvState]:
    """Collect trajectories with length of `rollout_length`.

    This method is a general rollout method used for collecting trajectories from a vectorized env. When the env enables autoreset, the returned sequential trajactory data could contain segments from multiple episodes.

    Args:
        env_fn: `step` function of a vmapped env.
        action_fn: The agent's action function. Eg: `agent.compute_actions`.
        env_state: State of the environment.
        agent_state: State of the agent.
        key: PRNG key.
        rollout_length: The length of the trajectory to collect.
        env_extra_fields: Extra fields collected from `env_state.info` into `trajectory.extras.env_extras`.

    Returns:
        A tuple (trajectory, env_state).
            - trajectory: `SampleBatch` object with shape (T, B, ...), where T=rollout_length, B=#envs in `env_fn`.
            - env_state: last env_state after rollout

    """

    def _one_step_rollout(carry, unused_t):
        env_state, current_key = carry
        next_key, current_key = rng_split(current_key, 2)

        # transition: [#envs, ...]
        transition, env_nstate = env_step(
            env_fn,
            action_fn,
            env_state,
            agent_state,
            current_key,
            env_extra_fields,
        )

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key), (), length=rollout_length
    )

    return trajectory, env_state


def eval_rollout_episode(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
) -> tuple[SampleBatch, EnvState]:
    """Evaulate a batch of episodic trajectories.

    It avoids unnecessary calls of `env_step()` when all environments are done. However, the agent's action function will still be called after that. When the function is wrapped by `jax.vmap()`, this mechanism will not work.

    Args:
        env_fn: `step` function of a vmapped env without autoreset.
        action_fn: The agent's action function. Eg: `agent.compute_actions`.
        env_state: State of the environment.
        agent_state: State of the agent.
        key: PRNG key.
        rollout_length: The length of the episodes. This value usually keeps the same as the env's `max_episode_steps` or be smllar than that.

    Returns:
        A tuple (trajectory, env_state).
            - trajectory: SampleBatch with shape (T, B, ...), where T=rollout_length, B=#envs. When a episode is terminated
            - env_state: last env_state after rollout
    """
    _eval_env_step = partial(eval_env_step, env_fn, action_fn)

    def _one_step_rollout(carry, unused_t):
        env_state, current_key, prev_transition = carry
        next_key, current_key = rng_split(current_key, 2)

        transition, env_nstate = jax.lax.cond(
            env_state.done.all(),
            lambda *x: (prev_transition.replace(), env_state.replace()),
            _eval_env_step,
            env_state,
            agent_state,
            current_key,
        )

        return (env_nstate, next_key, transition), transition

    # run one-step rollout first to get bootstrap transition
    # it will not include in the trajectory when env_state is from env.reset()
    # this is manually controlled by user.
    bootstrap_transition, _ = _eval_env_step(env_state, agent_state, key)

    (env_state, _, _), trajectory = jax.lax.scan(
        _one_step_rollout,
        (env_state, key, bootstrap_transition),
        (),
        length=rollout_length,
    )

    return trajectory, env_state


def fast_eval_rollout_episode(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
) -> tuple[PyTreeDict, EnvState]:
    """Fast evaulate a batch of episodic trajectories.

    A even faster implementation than `eval_rollout_episode()`. It achieves early termination when it is not wrapped by `jax.vmap()`. However, this method does not collect the trajectory data, it only returns the aggregated metrics dict with keys (episode_returns, episode_lengths), which is useful in cases like evaluation.

    Args:
        env_fn: `step` function of a vmapped env without autoreset.
        action_fn: The agent's action function. Eg: `agent.compute_actions`.
        env_state: State of the environment.
        agent_state: State of the agent.
        key: PRNG key.
        rollout_length: The length of the episodes. This value usually keeps the same as the env's `max_episode_steps`.

    Returns:
        metrics: Dict(episode_returns, episode_lengths)
        env_state: Last env_state after evaluation.
    """
    _eval_env_step = partial(eval_env_step, env_fn, action_fn)

    def _terminate_cond(carry):
        env_state, current_key, prev_metrics = carry
        return (prev_metrics.episode_lengths < rollout_length).all() & (
            ~env_state.done.all()
        )

    def _one_step_rollout(carry):
        env_state, current_key, prev_metrics = carry
        next_key, current_key = rng_split(current_key, 2)

        transition, env_nstate = _eval_env_step(env_state, agent_state, current_key)

        prev_dones = env_state.done

        metrics = PyTreeDict(
            episode_returns=prev_metrics.episode_returns
            + (1 - prev_dones) * transition.rewards,
            episode_lengths=prev_metrics.episode_lengths
            + (1 - prev_dones).astype(jnp.int32),
        )

        return env_nstate, next_key, metrics

    batch_shape = env_state.reward.shape

    env_state, _, metrics = jax.lax.while_loop(
        _terminate_cond,
        _one_step_rollout,
        (
            env_state,
            key,
            PyTreeDict(
                episode_returns=jnp.zeros(batch_shape),
                episode_lengths=jnp.zeros(batch_shape, dtype=jnp.int32),
            ),
        ),
    )

    return metrics, env_state
