import logging
import math
from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.agent import AgentActionFn
from evorl.envs import Env
from evorl.metrics import EvaluateMetric
from evorl.rollout import rollout, RolloutFn
from evorl.types import PyTreeNode, pytree_field
from evorl.sample_batch import SampleBatch
from evorl.utils.jax_utils import rng_split
from evorl.utils.rl_toolkits import compute_discount_return, compute_episode_length

logger = logging.getLogger(__name__)


class EpisodeCollector(PyTreeNode):
    """Evaluate and return eval metrics and episodic trajectory.

    EpisodeCollector is similar as `Evaluator`, but it also returns the trajectories. When evaluating the agent, call `rollout()`.

    Attributes:
        env: Vectorized environment w/o autoreset.
        action_fn: The agent action function.
        max_episode_steps: The maximum number of steps in an episode.
        env_extra_fields: The extra fields to collect from the environment.
        discount: The discount factor.
    """

    env: Env
    action_fn: AgentActionFn
    max_episode_steps: int = pytree_field(static=True)
    env_extra_fields: Sequence[str] = ()
    discount: float = 1.0

    rollout_fn: RolloutFn = pytree_field(default=rollout, static=True)

    def __post_init__(self):
        assert hasattr(self.env, "num_envs"), "only parrallel envs are supported"

    def rollout(
        self,
        agent_state,
        key: chex.PRNGKey,
        num_episodes: int,
    ) -> tuple[EvaluateMetric, SampleBatch]:
        num_envs = self.env.num_envs
        num_iters = math.ceil(num_episodes / num_envs)
        if num_episodes % num_envs != 0:
            logger.warning(
                f"num_episode ({num_episodes}) cannot be divided by parallel_envs ({num_envs}),"
                f"set new num_episodes={num_iters * num_envs}"
            )

        action_fn = self.action_fn
        env_reset_fn = self.env.reset
        env_step_fn = self.env.step

        def _evaluate_fn(key, unused_t):
            next_key, init_env_key = rng_split(key, 2)
            env_state = env_reset_fn(init_env_key)

            # Note: be careful when self.max_episode_steps < env.max_episode_steps,
            # where dones could all be zeros.
            episode_trajectory, env_state = self.rollout_fn(
                env_step_fn,
                action_fn,
                env_state,
                agent_state,
                key,
                self.max_episode_steps,
                self.env_extra_fields,
            )

            return next_key, episode_trajectory

        # [#iters, T, #envs]
        _, episode_trajectory = jax.lax.scan(_evaluate_fn, key, (), length=num_iters)

        # [#iters, T, #envs] -> [T, num_episodes]
        episode_trajectory = jtu.tree_map(
            lambda x: jax.lax.collapse(jnp.swapaxes(x, 0, 1), 1, 3), episode_trajectory
        )

        # [num_episodes]
        discount_returns = compute_discount_return(
            episode_trajectory.rewards, episode_trajectory.dones, self.discount
        )

        episode_lengths = compute_episode_length(episode_trajectory.dones)

        eval_metrics = EvaluateMetric(
            episode_returns=discount_returns,
            episode_lengths=episode_lengths,
        )

        return eval_metrics, episode_trajectory
