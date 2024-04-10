import jax
import jax.numpy as jnp
from evox import Problem, State
import chex
from typing import Tuple, Union

from evorl.agents import Agent, AgentState
from evorl.envs import create_wrapped_brax_env, Env, EnvState
from evorl.rollout import SampleBatch
from evorl.types import Reward, RewardDict, Callable, Action, PolicyExtraInfo, PyTreeDict
from evorl.utils.toolkits import compute_discount_return, compute_episode_length
import math
from functools import partial
import logging


logger = logging.getLogger(__name__)

# TODO: use dataclass after evox update


class MultiObjectiveBraxProblem(Problem):
    def __init__(
        self,
        agent: Agent,
        env_name: str,
        parallel_envs: int = 1,
        max_episode_steps: int = 1000,
        num_episodes: int = 10,
        discount: float = 1.0,
        metric_names: Tuple[str] = ('reward',),
        **env_kwargs
    ):
        """
            Args:
                agent: agent model that defined the weights
                env_name: name of the environment
                parallel_envs: number of parallel environments for a individual
                max_episode_steps: maximum number of steps in an episode
                num_episodes: number of episodes to evaluate,
                discount: discount factor for episode return calculation
                metric_names: names of the metrics to record as objectives.
                    By default, only original reward is recorded.
        """
        self.agent = agent
        self.max_episode_steps = max_episode_steps
        self.num_episodes = num_episodes
        self.discount = discount
        self.metric_names = metric_names

        self.num_iters = math.ceil(num_episodes / parallel_envs)
        if num_episodes % parallel_envs != 0:
            logger.warn(f"num_episode ({num_episodes}) cannot be divided by parallel_envs ({parallel_envs}),"
                        f"set new num_episodes={self.num_iters*parallel_envs}"
                        )

        self.env = create_wrapped_brax_env(
            env_name,
            episode_length=max_episode_steps,
            parallel=parallel_envs,
            autoreset=False,
            fast_reset=True,
            **env_kwargs
        )
        self.agent_eval_actions = jax.vmap(
            self.agent.evaluate_actions, axis_name='pop')
        # Note: there are two vmap: [#pop, #envs, ...]
        self.env_reset = jax.vmap(self.env.reset, axis_name='pop')
        self.env_step = jax.vmap(self.env.step, axis_name='pop')

    def setup(self, key: chex.PRNGKey):
        return State(
            key=key,
        )

    def evaluate(self, state: State, pop_agent_state: chex.ArrayTree) -> Tuple[chex.ArrayTree, State]:
        pop_size = jax.tree_leaves(pop_agent_state)[0].shape[0]

        def _evaluate_fn(key, unused_t):

            next_key, init_env_key = jax.random.split(key, 2)
            env_state = self.env_reset(
                jax.random.split(init_env_key, num=pop_size))

            env_state, episode_trajectory = eval_rollout_episode(
                self.env_step, self.agent, env_state, pop_agent_state,
                key, self.max_episode_steps, metric_names=self.metric_names
            )

            objectives = PyTreeDict()
            for name in episode_trajectory.rewards.keys():
                if 'reward' in name:
                    objectives[name] = compute_discount_return(
                        episode_trajectory.rewards[name],
                        episode_trajectory.dones,
                        self.discount
                    )
                else:
                    # For other metrics like 'x_position', we use the last value as the objective.
                    # Note: It is ok to use [-1], since wrapper ensures that the last value 
                    # repeats the terminal step value.
                    objectives[name] = episode_trajectory.rewards[name][-1]

            objectives.episode_lengths = compute_episode_length(
                episode_trajectory.dones)

            return next_key, objectives  # [#envs]

        # [#iters, #pop, #envs]
        key, objectives = jax.lax.scan(
            _evaluate_fn,
            state.key, (),
            length=self.num_iters)

        for k in objectives.keys():
            objectives[k] = jax.lax.collapse(
                jnp.swapaxes(objectives[k], 0, 1),
                1, 3
            ) # [#pop, num_episodes]

        return objectives, state.update(key=key)


def eval_env_step(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    sample_batch: SampleBatch,
    key: chex.PRNGKey,
    metric_names: Tuple[str] = (),
) -> Tuple[EnvState, SampleBatch]:

    actions, policy_extras = action_fn(agent_state, sample_batch, key)
    env_nstate = env_fn(env_state, actions)

    # info = env_nstate.info
    # env_extras = {x: info[x] for x in env_extra_fields if x in info}

    rewards = PyTreeDict({
        name: val
        for name, val in env_nstate.info.metrics.items()
        if name in metric_names
    })
    rewards.reward = env_nstate.reward

    transition = SampleBatch(
        rewards=rewards,
        dones=env_nstate.done,
    )

    return env_nstate, transition


def eval_rollout_episode(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    metric_names: Tuple[str] = (),
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect given rollout_length trajectory.
        Avoid unnecessary env_step()
        Args:
            env: vmapped env w/o autoreset
    """

    _eval_env_step = partial(eval_env_step, env_fn, action_fn, metric_names=metric_names)

    def _one_step_rollout(carry, unused_t):
        """
            sample_batch: one-step obs
            transition: one-step full info
        """
        env_state, current_key, prev_transition = carry
        next_key, current_key = jax.random.split(current_key, 2)

        sample_batch = SampleBatch(
            obs=env_state.obs,
        )

        env_nstate, transition = jax.lax.cond(
            env_state.done.all(),
            lambda *x: (env_state.replace(), prev_transition.replace()),
            _eval_env_step,
            env_state, agent_state,
            sample_batch, current_key
        )

        return (env_nstate, next_key, transition), transition

    # run one-step rollout first to get bootstrap transition
    # it will not include in the trajectory when env_state is from env.reset()
    # this is manually controlled by user.
    _, transition = _eval_env_step(
        env_state, agent_state,
        SampleBatch(obs=env_state.obs), key
    )

    (env_state, _, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key, transition),
        (), length=rollout_length
    )

    return env_state, trajectory
