import jax
import chex

from evorl.envs import Env
from evorl.agents import Agent

from evorl.rollout import eval_rollout_episode
from evorl.utils.toolkits import compute_discount_return
from flax import struct

@struct.dataclass
class EvaluateMetric:
    discount_return: chex.Array

@struct.dataclass
class Evaluator:
    env: Env
    agent: Agent
    max_episode_length: int
    discount: float = 1.0
    

    def evaluate(self, agent_state, num_episodes: int, key: chex.PRNGKey) -> EvaluateMetric:
        num_iters = num_episodes // self.env.num_envs

        def _evaluate_fn(key, unused_t):

            next_key, init_env_key = jax.random.split(key, 2)
            env_state = self.env.reset(init_env_key)
            env_state, trajectory = eval_rollout_episode(
                self.env, env_state, 
                self.agent, agent_state, 
                key, self.max_episode_length, 
                self.discount
            )

            discount_return = compute_discount_return(
                trajectory.rewards, trajectory.dones, self.discount)


            return next_key, discount_return  # [#envs]

        # [#iters, #envs]
        _, discount_return = jax.lax.scan(
            _evaluate_fn,
            key, (),
            length=num_iters)

        return EvaluateMetric(
            discount_return=discount_return.flatten() # [#iters * #envs]
            timesteps=
            )

