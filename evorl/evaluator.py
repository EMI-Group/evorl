import logging
import math

import chex
import jax
import jax.numpy as jnp

from evorl.agent import AgentState
from evorl.envs import Env
from evorl.metrics import EvaluateMetric
from evorl.rollout import eval_rollout_episode, fast_eval_rollout_episode
from evorl.types import PyTreeNode, pytree_field
from evorl.agent import AgentActionFn
from evorl.utils.jax_utils import rng_split
from evorl.utils.rl_toolkits import compute_discount_return, compute_episode_length

logger = logging.getLogger(__name__)


class Evaluator(PyTreeNode):
    env: Env = pytree_field(pytree_node=False)
    action_fn: AgentActionFn = pytree_field(pytree_node=False)
    max_episode_steps: int = pytree_field(pytree_node=False)
    discount: float = pytree_field(default=1.0, pytree_node=False)

    def evaluate(
        self, agent_state: AgentState, key: chex.PRNGKey, num_episodes: int
    ) -> EvaluateMetric:
        num_envs = self.env.num_envs
        num_iters = math.ceil(num_episodes / num_envs)
        if num_episodes % num_envs != 0:
            logger.warning(
                f"num_episode ({num_episodes}) cannot be divided by parallel_envs ({num_envs}),"
                f"set new num_episodes={num_iters*num_envs}"
            )

        action_fn = self.action_fn
        env_reset_fn = self.env.reset
        env_step_fn = self.env.step
        if key.ndim > 1:
            for _ in range(key.ndim - 1):
                action_fn = jax.vmap(action_fn)
                env_reset_fn = jax.vmap(env_reset_fn)
                env_step_fn = jax.vmap(env_step_fn)

        def _evaluate_fn(key, unused_t):
            next_key, init_env_key, eval_key = rng_split(key, 3)
            env_state = env_reset_fn(init_env_key)
            if self.discount == 1.0:
                episode_metrics, env_state = fast_eval_rollout_episode(
                    env_step_fn,
                    action_fn,
                    env_state,
                    agent_state,
                    eval_key,
                    self.max_episode_steps,
                )
                discount_returns = episode_metrics.episode_returns
                episode_lengths = episode_metrics.episode_lengths
            else:
                episode_trajectory, env_state = eval_rollout_episode(
                    env_step_fn,
                    action_fn,
                    env_state,
                    agent_state,
                    eval_key,
                    self.max_episode_steps,
                )

                # Note: be careful when self.max_episode_steps < env.max_episode_steps,
                # where dones could all be zeros.
                discount_returns = compute_discount_return(
                    episode_trajectory.rewards, episode_trajectory.dones, self.discount
                )
                episode_lengths = compute_episode_length(episode_trajectory.dones)

            return next_key, (discount_returns, episode_lengths)  # [..., #envs]

        # [#iters, ..., #envs]
        _, (discount_returns, episode_lengths) = jax.lax.scan(
            _evaluate_fn, key, (), length=num_iters
        )

        return EvaluateMetric(
            episode_returns=_flatten_metric(discount_returns),  # [..., num_episodes]
            episode_lengths=_flatten_metric(episode_lengths),
        )


def _flatten_metric(x):
    """
    x: (#iters, ..., #envs)

    Return: (..., #iters * #envs)
    """
    return jax.lax.collapse(jnp.moveaxis(x, 0, -2), -2)
