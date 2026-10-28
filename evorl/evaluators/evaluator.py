import logging
import math

import jax

from evorl.agent import AgentState
from evorl.envs import BaseEnv, Space
from evorl.metrics import EvaluateMetric
from evorl.rollout import (
    rollout,
    fast_eval_rollout_episode,
    host_fast_eval_rollout_episode,
)
from evorl.types import Action, PyTreeNode, pytree_field
from evorl.agent import AgentActionFn
from evorl.utils.jax_utils import rng_split
from evorl.utils.rl_toolkits import compute_discount_return, compute_episode_length

logger = logging.getLogger(__name__)


class Evaluator(PyTreeNode):
    """Evaluate the agent in the environments.

    Attributes:
        env: Vectorized environment w/o autoreset.
        action_fn: The agent action function.
        max_episode_steps: The maximum number of steps in an episode.
        discount: The discount factor.
    """

    env: BaseEnv[Action, Space] = pytree_field(static=True)
    action_fn: AgentActionFn = pytree_field(static=True)
    max_episode_steps: int = pytree_field(static=True)
    discount: float = pytree_field(default=1.0, static=True)

    def __post_init__(self):
        assert hasattr(self.env, "num_envs"), "only vectorized envs are supported"
        # assert self.max_episode_steps <= self.env.max_episode_steps, (
        #     f"max_episode_steps {self.max_episode_steps} should be equal or less than env.max_episode_steps {self.env.max_episode_steps}"
        # )

    def evaluate(
        self,
        agent_state: AgentState,
        key: jax.Array,
        num_episodes: int,
    ) -> EvaluateMetric:
        """Evaluate the agent based on its state.

        Args:
            agent_state: The state of the agent.
            key: The PRNG key.
            num_episodes: The number of episodes to evaluate.

        Returns:
            Evaluation metrics. For scalar rewards, `episode_return` has
            shape `(actual_num_episodes,)`; for vector rewards it has shape
            `(actual_num_episodes, num_objectives)`. `episode_length` always
            has shape `(actual_num_episodes,)`. Episode counts are rounded up
            to complete environment batches.
        """
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
        if getattr(self.env, "uses_host_callback", False):
            fast_rollout_fn = host_fast_eval_rollout_episode
        else:
            fast_rollout_fn = fast_eval_rollout_episode

        def _evaluate_fn(key, unused_t):
            next_key, init_env_key, eval_key = rng_split(key, 3)
            env_state = env_reset_fn(init_env_key)
            if self.discount == 1.0:
                episode_metrics, env_state = fast_rollout_fn(
                    env_step_fn,
                    action_fn,
                    env_state,
                    agent_state,
                    eval_key,
                    self.max_episode_steps,
                )
                episode_returns = episode_metrics.episode_return
                episode_lengths = episode_metrics.episode_length
            else:
                episode_trajectory, env_state = rollout(
                    env_step_fn,
                    action_fn,
                    env_state,
                    agent_state,
                    eval_key,
                    self.max_episode_steps,
                )

                # Note: be careful when self.max_episode_steps < env.max_episode_steps,
                # where dones could all be zeros.
                # compute_discount_return & compute_episode_length are fine!
                episode_returns = compute_discount_return(
                    episode_trajectory.rewards, episode_trajectory.dones, self.discount
                )
                episode_lengths = compute_episode_length(episode_trajectory.dones)

            return next_key, (episode_returns, episode_lengths)  # [..., #envs]

        # [#iters, #envs]
        _, (episode_returns, episode_lengths) = jax.lax.scan(
            _evaluate_fn, key, (), length=num_iters
        )

        # Merge evaluation batches and environments, preserving objective axes.
        eval_metrics = EvaluateMetric(
            episode_return=episode_returns.reshape((-1, *episode_returns.shape[2:])),
            episode_length=episode_lengths.flatten(),
        )

        return eval_metrics
