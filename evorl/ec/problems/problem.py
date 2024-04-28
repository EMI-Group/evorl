import jax
import jax.numpy as jnp
from evox import Problem, State
import chex
from typing import Tuple, Union, Callable, Sequence, Protocol

from evorl.agents import Agent, AgentState
from evorl.envs import EnvState, Env
from evorl.rollout import SampleBatch
from evorl.types import Action, PolicyExtraInfo, ReductionFn
from evorl.utils.toolkits import compute_discount_return, compute_episode_length
from evorl.utils.jax_utils import rng_split
import math
from functools import partial
import logging



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
            logger.warning(f"num_episode ({num_episodes}) cannot be divided by parallel envs ({parallel_envs}),"
                           f"set new num_episodes={self.num_iters*parallel_envs}"
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

            next_key, init_env_key, rollout_key = jax.random.split(key, 3)
            env_state = self.env_reset(
                jax.random.split(init_env_key, num=pop_size))

            env_state, episode_trajectory = eval_rollout_episode(
                self.env_step, self.agent_eval_actions,
                env_state, pop_agent_state,
                jax.random.split(rollout_key, num=pop_size),
                self.max_episode_steps
            )

            objectives = compute_discount_return(
                episode_trajectory.rewards,
                episode_trajectory.dones,
                self.discount
            )

            return next_key, objectives  # [#envs]

        # [#iters, #pop, #envs]
        key, objectives = jax.lax.scan(
            _evaluate_fn,
            state.key, (),
            length=self.num_iters)

        objectives = jax.lax.collapse(
            jnp.swapaxes(objectives, 0, 1),
            1, 3
        )  # [#pop, num_episodes]
        objectives = self.reduce_fn(objectives, axis=-1)

        return objectives, state.replace(key=key, objectives=objectives)


def eval_env_step(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    sample_batch: SampleBatch,
    key: chex.PRNGKey,
    # env_extra_fields: Tuple[str] = (),
) -> Tuple[EnvState, SampleBatch]:

    actions, policy_extras = action_fn(agent_state, sample_batch, key)
    env_nstate = env_fn(env_state, actions)

    # info = env_nstate.info
    # env_extras = {x: info[x] for x in env_extra_fields if x in info}

    transition = SampleBatch(
        rewards=env_nstate.reward,
        dones=env_nstate.done,
        # extras=
    )

    return env_nstate, transition


def eval_rollout_episode(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    # env_extra_fields: Tuple[str] = (),
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect given rollout_length trajectory.
        Avoid unnecessary env_step()
        Args:
            env: vmapped env w/o autoreset
    """

    _eval_env_step = partial(eval_env_step, env_fn, action_fn)

    def _one_step_rollout(carry, unused_t):
        """
            sample_batch: one-step obs
            transition: one-step full info
        """
        env_state, current_key, prev_transition = carry
        # next_key, current_key = jax.random.split(current_key, 2)
        next_key, current_key = rng_split(current_key, 2)

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
