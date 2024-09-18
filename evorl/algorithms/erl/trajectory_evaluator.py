import logging
import math
from collections.abc import Sequence

import chex
import jax
import jax.tree_util as jtu

from evorl.agent import AgentActionFn
from evorl.envs import EnvResetFn, EnvStepFn
from evorl.metrics import EvaluateMetric
from evorl.rollout import rollout
from evorl.types import PyTreeNode, pytree_field
from evorl.sample_batch import SampleBatch
from evorl.utils.jax_utils import rng_split
from evorl.utils.rl_toolkits import compute_discount_return, compute_episode_length

logger = logging.getLogger(__name__)


class TrajectoryEvaluator(PyTreeNode):
    """
    Return eval metrics and trajectory
    """

    env_step_fn: EnvStepFn
    env_reset_fn: EnvResetFn
    action_fn: AgentActionFn
    num_envs: int  # num_envs over env_step_fn
    max_episode_steps: int = pytree_field(pytree_node=False)
    env_extra_fields: Sequence[str] = ()

    def evaluate(
        self, agent_state, num_episodes: int, key: chex.PRNGKey
    ) -> tuple[EvaluateMetric, SampleBatch]:
        num_envs = self.num_envs
        num_iters = math.ceil(num_episodes / num_envs)
        if num_episodes % num_envs != 0:
            logger.warning(
                f"num_episode ({num_episodes}) cannot be divided by parallel_envs ({num_envs}),"
                f"set new num_episodes={num_iters*num_envs}"
            )

        return self._evaluate(agent_state, num_iters, key)

    def _evaluate(
        self, agent_state, num_iters: int, key: chex.PRNGKey
    ) -> tuple[EvaluateMetric, SampleBatch]:
        def _evaluate_fn(key, unused_t):
            next_key, init_env_key = rng_split(key, 2)
            env_state = self.env_reset_fn(init_env_key)

            # Note: be careful when self.max_episode_steps < env.max_episode_steps,
            # where dones could all be zeros.
            episode_trajectory, env_state = rollout(
                self.env_step_fn,
                self.action_fn,
                env_state,
                agent_state,
                key,
                self.max_episode_steps,
                self.env_extra_fields,
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
            episode_trajectory.rewards, episode_trajectory.dones
        )

        episode_lengths = compute_episode_length(episode_trajectory.dones)

        eval_metrics = EvaluateMetric(
            episode_returns=discount_returns,
            episode_lengths=episode_lengths,
        )

        return eval_metrics, episode_trajectory
