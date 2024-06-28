import jax
import jax.numpy as jnp
import chex
from typing import (
    Union, Tuple, Sequence, Callable
)
from .agents import Agent, AgentState
from .types import (
    Reward, RewardDict, Action, PolicyExtraInfo, PyTreeDict
)
from .sample_batch import SampleBatch, Episode
from .envs import Env, EnvState
from .utils.jax_utils import rng_split
from functools import partial

# TODO: add RNN Policy support


def env_step(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    sample_batch: SampleBatch,
    key: chex.PRNGKey,
    env_extra_fields: Sequence[str] = (),
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect one-step data.
    """

    actions, policy_extras = action_fn(agent_state, sample_batch, key)
    env_nstate = env_fn(env_state, actions)

    info = env_nstate.info
    env_extras = PyTreeDict(
        {x: info[x] for x in env_extra_fields if x in info})

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

def eval_env_step(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    sample_batch: SampleBatch,
    key: chex.PRNGKey
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect one-step data in evaluation mode.
    """

    actions, policy_extras = action_fn(agent_state, sample_batch, key)
    env_nstate = env_fn(env_state, actions)

    transition = SampleBatch(
        rewards=env_nstate.reward,
        dones=env_nstate.done,
    )

    return env_nstate, transition


def rollout(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    env_extra_fields: Sequence[str] = (),
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
        next_key, current_key = rng_split(current_key, 2)

        # sample_batch: [#envs, ...]
        sample_batch = SampleBatch(
            obs=env_state.obs
        )

        # transition: [#envs, ...]
        env_nstate, transition = env_step(
            env_fn, action_fn,
            env_state, agent_state,
            sample_batch, current_key, env_extra_fields
        )

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key), (),
        length=rollout_length
    )

    return env_state, trajectory


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

#     episodes = Episode(trajectory=trajectory, last_obs=env_state.obs)

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

#         env_nstate, transition = jax.lax.cond(
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
#     episodes = Episode(trajectory=trajectory, last_obs=env_state.obs)

#     return env_state, episodes




def eval_rollout(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
) -> Tuple[EnvState, Union[Reward, RewardDict]]:
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

        # sample_batch: [#envs, ...]
        sample_batch = SampleBatch(
            obs=env_state.obs,
        )

        # transition: [#envs, ...]
        env_nstate, transition = eval_env_step(
            env_fn, action_fn,
            env_state, agent_state,
            sample_batch, current_key
        )

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _), trajectory = jax.lax.scan(
        _one_step_rollout,
        (env_state, key), (),
        length=rollout_length
    )

    return env_state, trajectory


def eval_rollout_episode(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
) -> Tuple[EnvState, SampleBatch]:
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

        sample_batch = SampleBatch(
            obs=env_state.obs,
        )

        env_nstate, transition = jax.lax.cond(
            env_state.done.all(),
            lambda *x: (env_state.replace(), prev_transition.replace()),
            _eval_env_step,
            env_state, agent_state,
            sample_batch, current_key
        )

        return (env_nstate, next_key, transition), transition

    # run one-step rollout first to get bootstrap transition
    # it will not include in the trajectory when env_state is from env.reset()
    # this is manually controlled by user.
    _, bootstrap_transition = _eval_env_step(
        env_state, agent_state,
        SampleBatch(obs=env_state.obs), key
    )

    (env_state, _, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key, bootstrap_transition),
        (), length=rollout_length
    )

    return env_state, trajectory


def fast_eval_rollout_episode(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
) -> Tuple[EnvState, PyTreeDict]:
    """

        Args:
            env: vmapped env w/o autoreset
    """

    _eval_env_step = partial(eval_env_step, env_fn, action_fn)

    def _terminate_cond(carry):
        env_state, current_key, prev_metrics = carry
        return (prev_metrics.episode_lengths < rollout_length).all() & (~env_state.done.all())

    def _one_step_rollout(carry):
        """
            sample_batch: one-step obs
            transition: one-step full info
        """
        env_state, current_key, prev_metrics = carry
        next_key, current_key = rng_split(current_key, 2)

        sample_batch = SampleBatch(
            obs=env_state.obs,
        )

        env_nstate, transition = _eval_env_step(
            env_state, agent_state,
            sample_batch, current_key
        )

        metrics = PyTreeDict(
            episode_returns=prev_metrics.episode_returns +
            (1-transition.dones)*transition.rewards,
            episode_lengths=prev_metrics.episode_lengths +
            (1-transition.dones)
        )

        return env_nstate, next_key, metrics

    batch_shape = env_state.reward.shape

    env_state, _, metrics = jax.lax.while_loop(
        _terminate_cond,
        _one_step_rollout,
        (env_state, key,
         PyTreeDict(
             episode_returns=jnp.zeros(batch_shape),
             episode_lengths=jnp.zeros(batch_shape))
         )
    )

    return env_state, metrics
