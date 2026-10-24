"""Joint transitions for simultaneous-agent environments."""

import jax
import jax.numpy as jnp

from evorl.distributed.sharding import mark_varying
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict


def multi_agent_rollout(env, action_fn, env_state, agent_state, key, rollout_length):
    """Collect joint transitions, retaining original observations at boundaries.

    With NORMAL autoreset, a completed episode already exposes reset
    observations. With ENVPOOL autoreset, the next call only resets and every
    agent is inactive for that transition. An individually finished agent stays
    inactive until the whole episode resets.
    """

    def step(carry, _):
        state, key = carry
        key, action_key = jax.random.split(key)
        actions, extras = action_fn(agent_state, SampleBatch(obs=state.obs), action_key)
        next_state = env.step(state, actions)
        autoreset = jnp.asarray(next_state.info.get("autoreset", False), jnp.bool_)
        active = PyTreeDict(
            {
                name: (
                    (
                        ~state.done[name].astype(jnp.bool_)
                        | state.done["__all__"].astype(jnp.bool_)
                    )
                    & ~autoreset
                )
                for name in env.agents
            }
        )
        terminations = next_state.info.termination
        dones = PyTreeDict(
            {
                name: jnp.maximum(next_state.done[name], next_state.done["__all__"])
                for name in env.agents
            }
        )
        terminations = PyTreeDict(
            {
                name: jnp.maximum(terminations[name], terminations["__all__"])
                for name in env.agents
            }
        )
        transition = SampleBatch(
            obs=state.obs,
            actions=actions,
            rewards=PyTreeDict({name: next_state.reward[name] for name in env.agents}),
            next_obs=next_state.info.get("ori_obs", next_state.obs),
            dones=dones,
            extras=PyTreeDict(
                policy_extras=extras,
                active=active,
                terminations=terminations,
                episode_done=next_state.done["__all__"],
                global_obs=state.info.get("global_obs"),
                next_global_obs=next_state.info.get(
                    "ori_global_obs", next_state.info.get("global_obs")
                ),
            ),
        )
        return (next_state, key), transition

    axes = jax.sharding.get_abstract_mesh().manual_axes
    (env_state, _), trajectory = jax.lax.scan(
        step, (mark_varying(env_state, axes), key), (), length=rollout_length
    )
    return trajectory, env_state
