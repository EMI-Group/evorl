import logging
import math

import chex
import jax

from evorl.agent import Agent
from evorl.envs import Env
from evorl.metrics import EvaluateMetric
from evorl.rollout import eval_rollout_episode, fast_eval_rollout_episode
from evorl.types import PyTreeNode, pytree_field
from evorl.utils.rl_toolkits import compute_discount_return, compute_episode_length

logger = logging.getLogger(__name__)


class Evaluator(PyTreeNode):
    env: Env
    agent: Agent
    max_episode_steps: int = pytree_field(pytree_node=False)
    discount: float = pytree_field(default=1.0, pytree_node=False)

    def evaluate(
        self, agent_state, key: chex.PRNGKey, num_episodes: int
    ) -> EvaluateMetric:
        num_envs = self.env.num_envs
        num_iters = math.ceil(num_episodes / num_envs)
        if num_episodes % num_envs != 0:
            logger.warning(
                f"num_episode ({num_episodes}) cannot be divided by parallel_envs ({num_envs}),"
                f"set new num_episodes={num_iters*num_envs}"
            )

        if self.discount == 1.0:
            return self._fast_evaluate(agent_state, key, num_iters)
        else:
            return self._evaluate(agent_state, key, num_iters)

    def _evaluate(
        self, agent_state, key: chex.PRNGKey, num_iters: int
    ) -> EvaluateMetric:
        def _evaluate_fn(key, unused_t):
            next_key, init_env_key = jax.random.split(key, 2)
            env_state = self.env.reset(init_env_key)

            # Note: be careful when self.max_episode_steps < env.max_episode_steps,
            # where dones could all be zeros.
            episode_trajectory, env_state = eval_rollout_episode(
                self.env.step,
                self.agent.evaluate_actions,
                env_state,
                agent_state,
                key,
                self.max_episode_steps,
            )

            discount_returns = compute_discount_return(
                episode_trajectory.rewards, episode_trajectory.dones, self.discount
            )

            episode_lengths = compute_episode_length(episode_trajectory.dones)

            return next_key, (discount_returns, episode_lengths)  # [#envs]

        # [#iters, #envs]
        _, (discount_returns, episode_lengths) = jax.lax.scan(
            _evaluate_fn, key, (), length=num_iters
        )

        return EvaluateMetric(
            episode_returns=discount_returns.flatten(),  # [#iters * #envs]
            episode_lengths=episode_lengths.flatten(),
        )

    def _fast_evaluate(
        self, agent_state, key: chex.PRNGKey, num_iters: int
    ) -> EvaluateMetric:
        def _evaluate_fn(key, unused_t):
            next_key, init_env_key = jax.random.split(key, 2)
            env_state = self.env.reset(init_env_key)

            episode_metrics, env_state = fast_eval_rollout_episode(
                self.env.step,
                self.agent.evaluate_actions,
                env_state,
                agent_state,
                key,
                self.max_episode_steps,
            )

            return next_key, episode_metrics  # [#envs]

        # [#iters, #envs]
        _, episode_metrics = jax.lax.scan(_evaluate_fn, key, (), length=num_iters)

        return EvaluateMetric(
            episode_returns=episode_metrics.episode_returns.flatten(),  # [#iters * #envs]
            episode_lengths=episode_metrics.episode_lengths.flatten(),
        )
