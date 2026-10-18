import logging
import math
from typing import Any

import chex
import jax
import jax.tree_util as jtu

from evorl.agent import AgentState, AgentActionFn, RandomAgent
from evorl.distributed.sharding import mark_varying
from evorl.envs import BaseEnv, EnvState, EnvStepFn, create_env, AutoresetMode
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeNode
from evorl.utils.jax_utils import rng_split
from evorl.utils import running_statistics


logger = logging.getLogger(__name__)


class ObsPreprocessor(PyTreeNode):
    random_timesteps: int = 0
    static: bool = False  # set True means using VBN (eg: OpenES)

    def __post_init__(self):
        if self.static:
            assert self.random_timesteps > 0, (
                "init_timesteps should be greater than 0 if static is True"
            )


def init_obs_preprocessor(agent_state, config, key, dp_axis_name=None):
    """Initialize observation statistics from random rollouts.

    Rollout reads its Manual axes from the current `shard_map` context.
    `dp_axis_name` controls only statistics reduction; `None` keeps each
    workflow's statistics independent, including inside PBT.

    Args:
        agent_state: Agent state whose observation statistics are initialized.
        config: Configuration containing environment and warmup settings.
        key: PRNG key for environment, agent, and rollout initialization.
        dp_axis_name: Axis for statistics reduction, or `None` to disable it.

    Returns:
        An agent state with initialized observation preprocessing statistics.
    """
    assert config.random_timesteps > 0, "random_timesteps should be greater than 0"

    env = create_env(
        config.env,
        episode_length=config.env.max_episode_steps,
        parallel=config.num_envs,
        autoreset_mode=AutoresetMode.NORMAL,
    )

    obs_preprocessor_state = init_obs_preprocessor_with_random_timesteps(
        agent_state.obs_preprocessor_state,
        config.random_timesteps,
        env,
        key,
        dp_axis_name=dp_axis_name,
    )

    agent_state = agent_state.replace(obs_preprocessor_state=obs_preprocessor_state)

    return agent_state


def init_obs_preprocessor_with_random_timesteps(
    obs_preprocessor_state: Any,
    timesteps: int,
    env: BaseEnv,
    key: jax.Array,
    dp_axis_name: str | None = None,
) -> Any:
    """Collect observations and initialize their running statistics.

    Rollout obtains its VMA axes from the enclosing Manual context.
    `dp_axis_name` explicitly selects the statistics reduction axis;
    `None` disables statistics reduction.

    Args:
        obs_preprocessor_state: Running statistics state to update.
        timesteps: Positive observation count, rounded up to complete batches
            of environments.
        env: Vectorized environment used for random rollouts.
        key: PRNG key for environment, agent, and rollout initialization.
        dp_axis_name: Axis for statistics reduction, or `None` to disable it.

    Returns:
        The running statistics state updated with the collected observations.
    """
    env_key, agent_key, rollout_key = jax.random.split(key, num=3)
    env_state = env.reset(env_key)

    agent = RandomAgent()

    agent_state = agent.init(env.obs_space, env.action_space, agent_key)

    rollout_length = math.ceil(timesteps / env.num_envs)

    if rollout_length > 0:
        # obs (rollout_length, num_envs, ...)
        obs, env_state = rollout_obs(
            env.step,
            agent.compute_actions,
            env_state,
            agent_state,
            rollout_key,
            rollout_length=rollout_length,
        )

        obs = jtu.tree_map(lambda x: jax.lax.collapse(x, 0, 2), obs)

    obs_preprocessor_state = running_statistics.update(
        obs_preprocessor_state, obs, dp_axis_name=dp_axis_name
    )

    return obs_preprocessor_state


def rollout_obs(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,
    key: jax.Array,
    rollout_length: int,
) -> tuple[chex.ArrayTree, EnvState]:
    """Collect observations with VMA casts over the current Manual axes.

    Query the trace-time `shard_map` context, also under jit/vmap. Outside
    a Manual region the casts are a no-op. They perform no forward communication.

    Args:
        env_fn: Step function of a vectorized environment.
        action_fn: Agent action function.
        env_state: Initial environment state.
        agent_state: Agent state used to compute actions.
        key: PRNG key for action sampling.
        rollout_length: Number of environment steps to collect.

    Returns:
        A tuple `(obs, env_state)` containing the observations before each step
        with shape `(rollout_length, num_envs, ...)` and the final environment
        state.
    """
    manual_axes = jax.sharding.get_abstract_mesh().manual_axes

    def _one_step_rollout(carry, unused_t):
        env_state, current_key = carry
        next_key, current_key = rng_split(current_key, 2)
        sample_batch = SampleBatch(obs=env_state.obs)
        actions, policy_extras = action_fn(agent_state, sample_batch, current_key)
        env_nstate = env_fn(env_state, actions)

        return (env_nstate, next_key), env_state.obs  # obs_t

    # trajectory: [T, #envs, ...]
    (env_state, _), obs = jax.lax.scan(
        _one_step_rollout,
        (mark_varying(env_state, manual_axes), key),
        (),
        length=rollout_length,
    )

    return obs, env_state
