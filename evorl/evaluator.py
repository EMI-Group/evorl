import jax
import chex

from evorl.envs import Env
from evorl.agents import Agent
from evorl.metrics import EvaluateMetric
from evorl.rollout import eval_rollout_episode
from evorl.utils.toolkits import compute_discount_return, compute_episode_length

import dataclasses
import logging

import math

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Evaluator:
    env: Env
    agent: Agent
    max_episode_steps: int
    discount: float = 1.0
    # pmap_axis_name: Optional[str] = None

    # def enable_multi_devices(self, pmap_axis_name: Optional[str] = None):
    #     self.pmap_axis_name = pmap_axis_name

    def evaluate(self, agent_state, num_episodes: int, key: chex.PRNGKey) -> EvaluateMetric:
        num_envs = self.env.num_envs
        num_iters = math.ceil(num_episodes / num_envs)
        if num_episodes % num_envs != 0:
            logger.warn(f"num_episode ({num_episodes}) cannot be divided by parallel_envs ({num_envs}),"
                        f"set new num_episodes={self.num_iters*num_envs}"
                        )
        def _evaluate_fn(key, unused_t):

            next_key, init_env_key = jax.random.split(key, 2)
            env_state = self.env.reset(init_env_key)

            env_state, episode_trajectory = eval_rollout_episode(
                self.env, self.agent, env_state, agent_state,
                key, self.max_episode_steps
            )

            discount_returns = compute_discount_return(
                episode_trajectory.rewards, episode_trajectory.dones, self.discount)

            episode_lengths = compute_episode_length(episode_trajectory.dones)

            return next_key, (discount_returns, episode_lengths)  # [#envs]

        # [#iters, #envs]
        _, (discount_returns, episode_lengths) = jax.lax.scan(
            _evaluate_fn,
            key, (),
            length=num_iters)

        return EvaluateMetric(
            discount_returns=discount_returns.flatten(),  # [#iters * #envs]
            episode_lengths=episode_lengths.flatten()
        )
