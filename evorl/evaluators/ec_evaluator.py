import logging
import math
from typing import Any

import chex
import jax
import jax.tree_util as jtu

from evorl.agent import AgentState, AgentActionFn, RandomAgent
from evorl.envs import Env, EnvState, EnvStepFn
from evorl.sample_batch import SampleBatch
from evorl.metrics import EvaluateMetric
from evorl.types import (
    PyTreeData,
    PyTreeNode,
    pytree_field,
)
from evorl.utils.jax_utils import rng_split
from evorl.utils.rl_toolkits import compute_discount_return, compute_episode_length
from evorl.utils import running_statistics

logger = logging.getLogger(__name__)


class ObsPreprocessorSpec(PyTreeData):
    init_timesteps: int = 0
    static: bool = False  # set True means using VBN (eg: OpenES)

    def __post_init__(self):
        if self.static:
            assert (
                self.init_timesteps > 0
            ), "init_timesteps should be greater than 0 if static is True"


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

        obs = jax.lax.collapse(obs, 0, 2)

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
        """
        sample_batch: one-step obs
        transition: one-step full info
        """
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


def env_step(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    key: chex.PRNGKey,
) -> tuple[SampleBatch, EnvState]:
    """
    Collect one-step data.
    """
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
        )

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key), (), length=rollout_length
    )

    return trajectory, env_state


class EpisodeCollector(PyTreeNode):
    """
    Return eval metrics and trajectory
    """

    env: Env
    action_fn: AgentActionFn
    max_episode_steps: int = pytree_field(pytree_node=False)
    discount: float = 1.0

    def __post_init__(self):
        assert hasattr(self.env, "num_envs"), "only parrallel envs are supported"

    def rollout(
        self, agent_state, num_episodes: int, key: chex.PRNGKey
    ) -> tuple[EvaluateMetric, chex.ArrayTree]:
        num_envs = self.env.num_envs
        num_iters = math.ceil(num_episodes / num_envs)
        if num_episodes % num_envs != 0:
            logger.warning(
                f"num_episode ({num_episodes}) cannot be divided by parallel_envs ({num_envs}),"
                f"set new num_episodes={num_iters*num_envs}"
            )

        return self._evaluate(agent_state, num_iters, key)

    def _evaluate(
        self, agent_state, num_iters: int, key: chex.PRNGKey
    ) -> tuple[EvaluateMetric, chex.ArrayTree]:
        def _evaluate_fn(key, unused_t):
            next_key, init_env_key = rng_split(key, 2)
            env_state = self.env.reset(init_env_key)

            # Note: be careful when self.max_episode_steps < env.max_episode_steps,
            # where dones could all be zeros.
            episode_trajectory, env_state = rollout(
                self.env.step,
                self.action_fn,
                env_state,
                agent_state,
                key,
                self.max_episode_steps,
            )

            return next_key, episode_trajectory

        # [#iters, T, #envs, ...]
        _, episode_trajectory = jax.lax.scan(_evaluate_fn, key, (), length=num_iters)

        # [#iters, T, #envs] -> [T, #envs * #iters]
        episode_trajectory = jtu.tree_map(
            lambda x: jax.lax.collapse(x.swapaxes(0, 1), 1, 3), episode_trajectory
        )

        # [#envs * #iters]
        discount_returns = compute_discount_return(
            episode_trajectory.rewards, episode_trajectory.dones, self.discount
        )

        episode_lengths = compute_episode_length(episode_trajectory.dones)

        eval_metrics = EvaluateMetric(
            episode_returns=discount_returns,
            episode_lengths=episode_lengths,
        )

        return eval_metrics, episode_trajectory
