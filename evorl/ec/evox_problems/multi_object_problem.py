import logging
import copy

import chex
import jax
import jax.numpy as jnp
from evorl.agent import Agent, AgentStateAxis
from evorl.envs import Env
from evorl.types import ReductionFn, PyTreeDict
from evorl.evaluators import BraxEvaluator
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
        agent_state_vmap_axes: AgentStateAxis = 0,
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
        self.agent_state_vmap_axes = agent_state_vmap_axes

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

        metric_names = copy.deepcopy(self.metric_names)
        if "episode_length" not in metric_names:
            # we also need episode_length to calculate the sampled_timesteps
            metric_names = metric_names + ("episode_length",)

        self.evaluator = BraxEvaluator(
            env, action_fn, max_episode_steps, discount, metric_names
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

        raw_objectives = jax.vmap(
            self.evaluator.evaluate,
            in_axes=(self.agent_state_vmap_axes, 0, None),
        )(pop_agent_state, eval_key, self.num_episodes)

        sampled_timesteps = raw_objectives.episode_length.sum()
        sampled_episodes = jnp.uint32(pop_size * self.num_episodes)

        objectives = PyTreeDict(
            {
                k: self.reduce_fn[k](raw_objectives[k], axis=-1)
                for k in self.metric_names
            }
        )

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
