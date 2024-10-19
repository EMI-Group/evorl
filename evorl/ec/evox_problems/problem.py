import logging

import chex
import jax
import jax.numpy as jnp
from evorl.agent import Agent, AgentState, AgentStateAxis
from evorl.envs import Env
from evorl.evaluators import Evaluator
from evorl.types import ReductionFn
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
        explore: bool = False,
        reduce_fn: ReductionFn = jnp.mean,
        agent_state_vmap_axes: AgentStateAxis = 0,
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
        self.agent_state_vmap_axes = agent_state_vmap_axes

        if explore:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        self.evaluator = Evaluator(env, action_fn, max_episode_steps, discount)

    @property
    def num_objectives(self):
        return 1

    def setup(self, key: chex.PRNGKey):
        return State(
            key=key,
            # sampled timesteps at each iteration
            sampled_timesteps=jnp.zeros((), dtype=jnp.uint32),
            sampled_episodes=jnp.zeros((), dtype=jnp.uint32),
        )

    def evaluate(
        self, state: State, pop_agent_state: AgentState
    ) -> tuple[chex.ArrayTree, State]:
        pop_size = jax.tree_leaves(pop_agent_state)[0].shape[0]

        key, eval_key = jax.random.split(state.key)
        eval_key = jax.random.split(eval_key, num=pop_size)  # [#pop]

        #  [#pop, num_episodes]
        eval_metrics = jax.vmap(
            self.evaluator.evaluate,
            in_axes=(self.agent_state_vmap_axes, 0, None),
        )(pop_agent_state, eval_key, self.num_episodes)

        objectives = eval_metrics.episode_returns
        sampled_episodes = jnp.uint32(pop_size * self.num_episodes)
        sampled_timesteps = eval_metrics.episode_lengths.sum().astype(jnp.uint32)

        # by default, we use the mean value over different episodes.
        objectives = self.reduce_fn(objectives, axis=-1)

        state = state.replace(
            key=key,
            sampled_timesteps=sampled_timesteps,
            sampled_episodes=sampled_episodes,
        )

        return objectives, state
