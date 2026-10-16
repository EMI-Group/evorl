from collections.abc import Sequence
from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp

from evorl.agent import AgentActionFn, AgentState
from evorl.distributed.sharding import mark_varying
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
        key: jax.Array,
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
    key: jax.Array,
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
    key: jax.Array,
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
    key: jax.Array,
    rollout_length: int,
    env_extra_fields: Sequence[str] = (),
) -> tuple[SampleBatch, EnvState]:
    """Collect fixed-length trajectories from a vectorized environment.

    With autoreset enabled, the trajectory can contain segments from multiple
    episodes.

    Inside `shard_map`, read the current Manual axes at trace time and mark
    the initial loop state as varying over them. This also works under jit/vmap,
    requires no axis argument from the caller, and performs no forward
    communication. Outside a Manual region the casts are a no-op.

    Args:
        env_fn: Step function of a vectorized environment.
        action_fn: Agent action function, such as `agent.compute_actions`.
        env_state: State of the environment.
        agent_state: State of the agent.
        key: PRNG key.
        rollout_length: The length of the trajectory to collect.
        env_extra_fields: Fields copied from `env_state.info` to
            `trajectory.extras.env_extras`.

    Returns:
        A tuple `(trajectory, env_state)` containing the transitions with shape
        `(rollout_length, num_envs, ...)` and the final environment state.
    """
    # Query the trace-time shard_map context; this also works under jit/vmap.
    # An empty tuple makes local execution a no-op. This performs no communication.
    manual_axes = jax.sharding.get_abstract_mesh().manual_axes

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
        _one_step_rollout,
        (mark_varying(env_state, manual_axes), key),
        (),
        length=rollout_length,
    )

    return trajectory, env_state


def eval_rollout_episode(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,
    key: jax.Array,
    rollout_length: int,
) -> tuple[SampleBatch, EnvState]:
    """Collect a fixed-length batch of episodic evaluation trajectories.

    Once all environments are done, later iterations reuse the previous
    transition and environment state. Under `jax.vmap`, the conditional may
    still execute the environment step for completed evaluations.

    VMA casts use the current Manual axes from the trace-time `shard_map`
    context, including under jit/vmap. They perform no forward communication
    and are a no-op outside a Manual region.

    Args:
        env_fn: Step function of a vectorized environment without autoreset.
        action_fn: Agent action function, such as `agent.compute_actions`.
        env_state: State of the environment.
        agent_state: State of the agent.
        key: PRNG key.
        rollout_length: Number of transitions to collect. Usually equal to or
            smaller than the environment's `max_episode_steps`.

    Returns:
        A tuple `(trajectory, env_state)` containing the evaluation transitions
        with shape `(rollout_length, num_envs, ...)` and the final environment
        state.
    """
    _eval_env_step = partial(eval_env_step, env_fn, action_fn)
    manual_axes = jax.sharding.get_abstract_mesh().manual_axes

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
    bootstrap_transition = mark_varying(bootstrap_transition, manual_axes)

    (env_state, _, _), trajectory = jax.lax.scan(
        _one_step_rollout,
        (mark_varying(env_state, manual_axes), key, bootstrap_transition),
        (),
        length=rollout_length,
    )

    return trajectory, env_state


def fast_eval_rollout_episode(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,
    key: jax.Array,
    rollout_length: int,
) -> tuple[PyTreeDict, EnvState]:
    """Evaluate episode returns and lengths without storing trajectories.

    Stop when all environments are done or the rollout length limit is reached.
    Under `jax.vmap`, batched iterations may continue for other evaluations.

    Mark the initial environment state and zero counters over the current
    Manual axes from the trace-time `shard_map` context, also under jit/vmap.
    The casts perform no forward communication and are a no-op outside a
    Manual region.

    Args:
        env_fn: Step function of a vectorized environment without autoreset.
        action_fn: Agent action function, such as `agent.compute_actions`.
        env_state: State of the environment.
        agent_state: State of the agent.
        key: PRNG key.
        rollout_length: Maximum episode length, usually the environment's
            `max_episode_steps`.

    Returns:
        A tuple `(metrics, env_state)`. The metrics contain `episode_returns`
        and `episode_lengths` for each environment; `env_state` is the final
        environment state.
    """
    _eval_env_step = partial(eval_env_step, env_fn, action_fn)
    manual_axes = jax.sharding.get_abstract_mesh().manual_axes

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
            mark_varying(env_state, manual_axes),
            key,
            mark_varying(
                PyTreeDict(
                    episode_returns=jnp.zeros(batch_shape),
                    episode_lengths=jnp.zeros(batch_shape, dtype=jnp.int32),
                ),
                manual_axes,
            ),
        ),
    )

    return metrics, env_state
