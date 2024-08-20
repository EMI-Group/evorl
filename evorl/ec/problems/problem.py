import logging
import math

import chex
import jax
import jax.numpy as jnp
from evorl.agents import Agent
from evorl.envs import Env
from evorl.rollout import eval_rollout_episode, fast_eval_rollout_episode
from evorl.types import ReductionFn
from evorl.utils.rl_toolkits import compute_discount_return, compute_episode_length
from evox import Problem, State

logger = logging.getLogger(__name__)

# TODO: use dataclass after evox update


class GeneralRLProblem(Problem):
    def __init__(
        self,
        agent: Agent,
        env: Env,
        num_episodes: int = 10,
        max_episode_steps: int = 1000,
        discount: float = 1.0,
        reduce_fn: ReductionFn = jnp.mean,
    ):
        """
        RL Problem wrapper for general RL problems. The objective is the discounted return.

        Args:
            agent: agent model that defined the weights
            env_name: name of the environment
            num_episodes: number of episodes to evaluate
            max_episode_steps: maximum steps for each episode
            discount: discount factor for episode return calculation
            reduce_fn: function or function list to reduce each objective over episodes.
        """
        self.agent = agent
        self.env = env

        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.discount = discount
        self.reduce_fn = reduce_fn

        parallel_envs = env.num_envs
        self.num_iters = math.ceil(num_episodes / parallel_envs)
        if num_episodes % parallel_envs != 0:
            logger.warning(
                f"num_episode ({num_episodes}) cannot be divided by parallel envs ({parallel_envs}),"
                f"set new num_episodes={self.num_iters*parallel_envs}"
            )

        self.agent_eval_actions = jax.vmap(self.agent.evaluate_actions)
        # Note: there are two vmap: [#pop, #envs, ...]
        self.env_reset = jax.vmap(self.env.reset)
        self.env_step = jax.vmap(self.env.step)

    def setup(self, key: chex.PRNGKey):
        return State(
            key=key,
            # sampled timesteps at each iteration
            sampled_timesteps=jnp.zeros((), dtype=jnp.uint32),
            sampled_episodes=jnp.zeros((), dtype=jnp.uint32),
        )

    def evaluate(
        self, state: State, pop_agent_state: chex.ArrayTree
    ) -> tuple[chex.ArrayTree, State]:
        pop_size = jax.tree_leaves(pop_agent_state)[0].shape[0]

        def _evaluate_fn(key, unused_t):
            next_key, init_env_key, rollout_key = jax.random.split(key, 3)
            env_state = self.env_reset(jax.random.split(init_env_key, num=pop_size))

            if self.discount == 1.0:
                # use fast undiscount evaluation
                episode_metrics, env_state = fast_eval_rollout_episode(
                    self.env_step,
                    self.agent_eval_actions,
                    env_state,
                    pop_agent_state,
                    jax.random.split(rollout_key, num=pop_size),
                    self.max_episode_steps,
                )

                objectives = episode_metrics.episode_returns
                sampled_timesteps = episode_metrics.episode_lengths
            else:
                episode_trajectory, env_state = eval_rollout_episode(
                    self.env_step,
                    self.agent_eval_actions,
                    env_state,
                    pop_agent_state,
                    jax.random.split(rollout_key, num=pop_size),
                    self.max_episode_steps,
                )

                objectives = compute_discount_return(
                    episode_trajectory.rewards, episode_trajectory.dones, self.discount
                )
                sampled_timesteps = compute_episode_length(episode_trajectory.dones)

            return next_key, (objectives, sampled_timesteps)  # [#envs]

        # [#iters, #pop, #envs]
        key, (objectives, sampled_timesteps) = jax.lax.scan(
            _evaluate_fn, state.key, (), length=self.num_iters
        )

        objectives = jax.lax.collapse(
            jnp.swapaxes(objectives, 0, 1), 1, 3
        )  # [#pop, num_episodes]
        # by default, we use the mean value over different episodes.
        objectives = self.reduce_fn(objectives, axis=-1)

        sampled_episodes = jnp.array(
            pop_size * self.num_episodes * self.env.num_envs, dtype=jnp.uint32
        )
        sampled_timesteps = sampled_timesteps.sum().astype(jnp.uint32)

        state = state.replace(
            key=key,
            sampled_timesteps=sampled_timesteps,
            sampled_episodes=sampled_episodes,
        )

        return objectives, state
