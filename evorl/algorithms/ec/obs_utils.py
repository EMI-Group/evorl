import logging
import math
from typing import Any

import chex
import jax
import jax.tree_util as jtu

from evorl.agent import AgentState, AgentActionFn, RandomAgent
from evorl.envs import Env, EnvState, EnvStepFn, create_env, AutoresetMode
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


def init_obs_preprocessor(agent_state, config, key, pmap_axis_name=None):
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
        pmap_axis_name=pmap_axis_name,
    )

    agent_state = agent_state.replace(obs_preprocessor_state=obs_preprocessor_state)

    return agent_state


def init_obs_preprocessor_with_random_timesteps(
    obs_preprocessor_state: Any,
    timesteps: int,
    env: Env,
    key: chex.PRNGKey,
    pmap_axis_name: str | None = None,
) -> Any:
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
        obs_preprocessor_state, obs, pmap_axis_name=pmap_axis_name
    )

    return obs_preprocessor_state


def rollout_obs(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
) -> tuple[chex.ArrayTree, EnvState]:
    def _one_step_rollout(carry, unused_t):
        env_state, current_key = carry
        next_key, current_key = rng_split(current_key, 2)
        sample_batch = SampleBatch(obs=env_state.obs)
        actions, policy_extras = action_fn(agent_state, sample_batch, current_key)
        env_nstate = env_fn(env_state, actions)

        return (env_nstate, next_key), env_state.obs  # obs_t

    # trajectory: [T, #envs, ...]
    (env_state, _), obs = jax.lax.scan(
        _one_step_rollout, (env_state, key), (), length=rollout_length
    )

    return obs, env_state
