import logging

import chex
import jax
import jax.numpy as jnp
from evorl.agent import Agent
from evorl.envs import Env
from evorl.types import ReductionFn
from evorl.mo_brax_evaluator import BraxEvaluator
from evox import Problem, State

logger = logging.getLogger(__name__)


class MultiObjectiveBraxProblem(Problem):
    def __init__(
        self,
        agent: Agent,
        env: Env,
        num_episodes: int = 10,
        max_episode_steps: int = 1000,
        discount: float = 1.0,
        metric_names: tuple[str] = ("reward", "episode_length"),
        flatten_objectives: bool = True,
        explore: bool = False,
        reduce_fn: ReductionFn | dict[str, ReductionFn] = jnp.mean,
    ):
        """
        Args:
            agent: agent model that defined the weights
            env_name: name of the environment
            num_episodes: number of episodes to evaluate
            max_episode_steps: maximum steps for each episode
            discount: discount factor for episode return calculation
            metric_names: names of the metrics to record as objectives.
            flatten_objectives: whether flatten the objectives or keep the dict structure.
            reduce_fn: function or function dict to reduce each objective over episodes.
        """
        self.agent = agent
        self.env = env

        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.discount = discount
        self.metric_names = tuple(metric_names)
        self.flatten_objectives = flatten_objectives

        if isinstance(reduce_fn, dict):
            assert (
                len(reduce_fn) == len(metric_names)
            ), "when reduce_fn is a list, it should have the same length as metric_names"
            self.reduce_fn = reduce_fn
        else:
            self.reduce_fn = {name: reduce_fn for name in metric_names}

        if explore:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        self.evaluator = BraxEvaluator(
            env, action_fn, max_episode_steps, discount, self.metric_names
        )

    @property
    def num_objectives(self):
        return len(self.metric_names)

    def setup(self, key: chex.PRNGKey):
        return State(
            key=key,
            # note: we use float32, as overflow may easily happens with int32 in EC
            sampled_timesteps=jnp.zeros(()),
            sampled_episodes=jnp.zeros(()),
        )

    def evaluate(
        self, state: State, pop_agent_state: chex.ArrayTree
    ) -> tuple[chex.ArrayTree, State]:
        pop_size = jax.tree_leaves(pop_agent_state)[0].shape[0]

        key, eval_key = jax.random.split(state.key)
        eval_key = jax.random.split(eval_key, num=pop_size)  # [#pop]

        objectives = self.evaluator.evaluate(
            pop_agent_state, eval_key, self.num_episodes
        )

        sampled_timesteps = objectives.episode_length.sum()
        sampled_episodes = jnp.uint32(pop_size * self.num_episodes * self.env.num_envs)

        for k in objectives.keys():
            objectives[k] = self.reduce_fn[k](objectives[k], axis=-1)

        if self.flatten_objectives:
            # [#pop, #objs]
            # TODO: check key orders
            objectives = jnp.stack(list(objectives.values()), axis=-1)
            # special handling for single objective
            if objectives.shape[-1] == 1:
                objectives = objectives.squeeze(-1)

        state = state.replace(
            key=key,
            sampled_timesteps=sampled_timesteps,
            sampled_episodes=sampled_episodes,
        )

        return objectives, state
