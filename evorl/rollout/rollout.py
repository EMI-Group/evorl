from collections.abc import Sequence
from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

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
    "host_fast_eval_rollout_episode",
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

    Supports both native JAX environments and host-side environments, such as
    Gymnasium and EnvPool, whose adapters execute environment steps through
    `jax.experimental.io_callback(ordered=True)`.

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
        Scalar rewards have shape `(rollout_length, num_envs)`; vector rewards
        retain a final objective axis, including for a single objective.
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
        A tuple `(metrics, env_state)`. The metrics contain `episode_return`
        and `episode_length` for each environment; `env_state` is the final
        environment state. Returns retain the reward's optional objective
        axis; lengths have the same shape as `env_state.done`.
    """
    _eval_env_step = partial(eval_env_step, env_fn, action_fn)
    manual_axes = jax.sharding.get_abstract_mesh().manual_axes

    def _terminate_cond(carry):
        env_state, current_key, prev_metrics = carry
        return (prev_metrics.episode_length < rollout_length).all() & (
            ~env_state.done.all()
        )

    def _one_step_rollout(carry):
        env_state, current_key, prev_metrics = carry
        next_key, current_key = rng_split(current_key, 2)

        transition, env_nstate = _eval_env_step(env_state, agent_state, current_key)

        prev_dones = env_state.done
        active = 1 - prev_dones
        # A vector reward adds an objective axis after the environment axes.
        reward_mask = active
        if transition.rewards.ndim == prev_dones.ndim + 1:
            reward_mask = reward_mask[..., None]

        metrics = PyTreeDict(
            episode_return=prev_metrics.episode_return
            + reward_mask * transition.rewards,
            episode_length=prev_metrics.episode_length + active.astype(jnp.int32),
        )

        return env_nstate, next_key, metrics

    env_state, _, metrics = jax.lax.while_loop(
        _terminate_cond,
        _one_step_rollout,
        (
            mark_varying(env_state, manual_axes),
            key,
            mark_varying(
                PyTreeDict(
                    episode_return=jnp.zeros_like(env_state.reward),
                    episode_length=jnp.zeros_like(env_state.done, dtype=jnp.int32),
                ),
                manual_axes,
            ),
        ),
    )

    return metrics, env_state


def host_fast_eval_rollout_episode(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,
    key: jax.Array,
    rollout_length: int,
) -> tuple[PyTreeDict, EnvState]:
    """Evaluate a batch of Gymnasium or EnvPool episodes with early stopping.

    Each iteration computes actions and advances the whole native environment
    pool through its ordered IO callback. The environment must preserve each
    completed episode's state, as the adapters' `OneEpisodeWrapper` does.
    Returns and lengths stop accumulating after each environment's first done.

    Under nested `vmap`, batch the environment and policy steps before building
    the shared while loop, which JAX cannot directly batch with ordered IO
    effects. Shared policy leaves and action keys remain unmapped. Reset and
    environment-pool management remain the caller's responsibility.

    Args:
        env_fn: Vectorized host environment step with autoreset disabled.
        action_fn: Agent action function, such as `agent.evaluate_actions`.
        env_state: Environment state returned by reset.
        agent_state: Read-only agent state.
        key: PRNG key for action sampling.
        rollout_length: Maximum number of steps per episode.

    Returns:
        A tuple `(metrics, env_state)` containing `episode_return`,
        `episode_length`, and the final environment state.
    """

    @custom_vmap
    def evaluate(env_state, agent_state, key):
        manual_axes = jax.sharding.get_abstract_mesh().manual_axes
        metrics = PyTreeDict(
            episode_return=jnp.zeros_like(env_state.reward),
            episode_length=jnp.zeros_like(env_state.done, dtype=jnp.int32),
        )

        def continue_evaluation(carry):
            state, key, metrics = carry
            return (metrics.episode_length.max() < rollout_length) & ~state.done.all()

        def step(carry):
            state, key, metrics = carry
            next_key, action_key = rng_split(key, 2)
            actions, _ = action_fn(agent_state, SampleBatch(obs=state.obs), action_key)
            next_state = env_fn(state, actions)
            # Include the terminal step's reward, then ignore later pool steps.
            active = ~state.done.astype(jnp.bool_)
            reward_mask = active
            if next_state.reward.ndim == state.done.ndim + 1:
                reward_mask = reward_mask[..., None]
            metrics = metrics.replace(
                episode_return=metrics.episode_return
                + jnp.where(reward_mask, next_state.reward, 0),
                episode_length=metrics.episode_length + active.astype(jnp.int32),
            )
            return next_state, next_key, metrics

        env_state, _, metrics = jax.lax.while_loop(
            continue_evaluation,
            step,
            (
                mark_varying(env_state, manual_axes),
                key,
                mark_varying(metrics, manual_axes),
            ),
        )
        return metrics, env_state

    @evaluate.def_vmap
    def batch(axis_size, in_batched, env_state, agent_state, key):
        # Environment state needs a separate slot for every evaluation. Keep
        # shared policy leaves and keys unmapped when batching the action function.
        env_state = jax.tree.map(
            lambda x, mapped: (
                x if mapped else jnp.broadcast_to(x, (axis_size, *jnp.shape(x)))
            ),
            env_state,
            in_batched[0],
        )
        agent_axes = jax.tree.map(lambda mapped: 0 if mapped else None, in_batched[1])
        key_axis = 0 if in_batched[2] else None
        # Re-enter this function so an outer vmap also batches before the loop.
        result = host_fast_eval_rollout_episode(
            jax.vmap(env_fn),
            jax.vmap(action_fn, in_axes=(agent_axes, 0, key_axis)),
            env_state,
            agent_state,
            key,
            rollout_length,
        )
        return result, jax.tree.map(lambda _: True, result)

    return evaluate(env_state, agent_state, key)
