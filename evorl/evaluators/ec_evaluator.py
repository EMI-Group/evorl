from collections.abc import Sequence

import chex
import jax

from evorl.agent import AgentState, AgentActionFn
from evorl.envs import EnvState, EnvStepFn
from evorl.sample_batch import SampleBatch
from evorl.types import pytree_field
from evorl.utils.jax_utils import rng_split

from .episode_collector import EpisodeCollector, RolloutFn


def env_step(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    key: chex.PRNGKey,
) -> tuple[SampleBatch, EnvState]:
    # sample_batch: [#envs, ...]
    sample_batch = SampleBatch(obs=env_state.obs)

    actions, policy_extras = action_fn(agent_state, sample_batch, key)
    env_nstate = env_fn(env_state, actions)

    transition = SampleBatch(
        obs=env_state.obs,
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
        )

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key), (), length=rollout_length
    )

    return trajectory, env_state


class EpisodeObsCollector(EpisodeCollector):
    """Streamlined episode collector for observation only."""

    rollout_fn: RolloutFn = pytree_field(default=rollout, static=True)
