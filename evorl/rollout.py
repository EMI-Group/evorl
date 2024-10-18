from collections.abc import Sequence
from functools import partial
from typing import Protocol

import chex
import jax
import jax.numpy as jnp

from evorl.agent import AgentActionFn, AgentState
from evorl.envs import EnvState, EnvStepFn
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict, Reward, RewardDict
from evorl.utils.jax_utils import rng_split

# TODO: add RNN Policy support


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
    """
    Collect one-step data.
    """
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
    """
    Collect one-step data in evaluation mode.
    """
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


# def rollout_episode(
#     env_fn: Callable[[EnvState, Action], EnvState],
#     action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
#     env_state: EnvState,
#     agent_state: AgentState,
#     key: chex.PRNGKey,
#     rollout_length: int,
#     env_extra_fields: Sequence[str] = ()
# ) -> Tuple[EnvState, Episode]:
#     """
#         Collect given rollout_length trajectory.
#         Args:
#             env: vmapped env w/o autoreset
#     """

#     env_state, trajectory = rollout(
#         env_fn, action_fn, env_state, agent_state, key, rollout_length, env_extra_fields
#     )

#     episodes = Episode(trajectory=trajectory, ori_obs=env_state.obs)

#     return env_state, episodes


# def fast_rollout_episode(
#     env_fn: Callable[[EnvState, Action], EnvState],
#     action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
#     env_state: EnvState,
#     agent_state: AgentState,
#     key: chex.PRNGKey,
#     rollout_length: int,
#     env_extra_fields: Sequence[str] = ()
# ) -> Tuple[EnvState, Episode]:
#     """
#         Collect given rollout_length trajectory.
#         Avoid unnecessary env_step()

#         This method is more efficient than rollout_episode() if
#         the terminated_steps << rollout_length. But it is a little
#         bit slower if the terminated_steps ~ rollout_length.

#         Args:
#             env: vmapped env w/o autoreset
#     """

#     _env_step = partial(env_step, env_fn, action_fn,
#                         env_extra_fields=env_extra_fields)

#     def _one_step_rollout(carry, unused_t):
#         """
#             sample_batch: one-step obs
#             transition: one-step full info
#         """
#         env_state, current_key, last_transition = carry
#         next_key, current_key = rng_split(current_key, 2)

#         sample_batch = SampleBatch(
#             obs=env_state.obs,
#         )

#         transition, env_nstate = jax.lax.cond(
#             env_state.done.all(),
#             lambda *x: (env_state.replace(), last_transition.replace()),
#             _env_step,
#             env_state, agent_state,
#             sample_batch, current_key,
#         )

#         return (env_nstate, next_key, transition), transition

#     # run one-step rollout first to get bootstrap transition
#     _, bootstrap_transition = _env_step(
#         env_state, agent_state,
#         SampleBatch(obs=env_state.obs), key
#     )

#     (env_state, _, _), trajectory = jax.lax.scan(
#         _one_step_rollout, (env_state, key, bootstrap_transition),
#         (), length=rollout_length
#     )

#     # valid_mask is still ensured
#     episodes = Episode(trajectory=trajectory, ori_obs=env_state.obs)

#     return env_state, episodes


def eval_rollout(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
) -> tuple[EnvState, Reward | RewardDict]:
    """
    Collect given rollout_length trajectory.

    Args:
        env: vmapped env w/o autoreset
        discount: discount factor. When discount=1.0, return undiscounted return.

    Returns:
        env_state: last env_state after rollout
        trajectory: shape: [T, #envs, ...]
    """

    def _one_step_rollout(carry, unused_t):
        """
        sample_batch: one-step obs
        transition: one-step full info
        """
        env_state, current_key = carry
        next_key, current_key = rng_split(current_key, 2)

        # transition: [#envs, ...]
        transition, env_nstate = eval_env_step(
            env_fn, action_fn, env_state, agent_state, current_key
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
    """
    Collect given rollout_length trajectory.
    Avoid unnecessary env_step()
    Args:
        env: vmapped env w/o autoreset
    """

    _eval_env_step = partial(eval_env_step, env_fn, action_fn)

    def _one_step_rollout(carry, unused_t):
        """
        sample_batch: one-step obs
        transition: one-step full info
        """
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
    """

    Args:
        env: vmapped env w/o autoreset
    """

    _eval_env_step = partial(eval_env_step, env_fn, action_fn)

    def _terminate_cond(carry):
        env_state, current_key, prev_metrics = carry
        return (prev_metrics.episode_lengths < rollout_length).all() & (
            ~env_state.done.all()
        )

    def _one_step_rollout(carry):
        """
        sample_batch: one-step obs
        transition: one-step full info
        """
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
