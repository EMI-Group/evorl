import copy
import logging
import math
from collections.abc import Callable
from functools import partial

import chex
import jax
import jax.numpy as jnp
from evorl.algorithms import Agent, AgentState
from evorl.envs import Env, EnvState
from evorl.rollout import SampleBatch
from evorl.types import Action, PolicyExtraInfo, PyTreeDict, ReductionFn
from evorl.utils.jax_utils import rng_split
from evorl.utils.rl_toolkits import compute_discount_return, compute_episode_length
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
        metric_names: tuple[str] = ("reward",),
        flatten_objectives: bool = True,
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
                By default, only original reward is recorded.
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
            # note: we use float32, as overflow may easily happens with int32 in EC
            sampled_timesteps=jnp.zeros(()),
            sampled_episodes=jnp.zeros(()),
        )

    def evaluate(
        self, state: State, pop_agent_state: chex.ArrayTree
    ) -> tuple[chex.ArrayTree, State]:
        pop_size = jax.tree_leaves(pop_agent_state)[0].shape[0]

        metric_names = copy.copy(self.metric_names)
        if "episode_length" not in metric_names:
            metric_names = metric_names + ("episode_length",)

        def _evaluate_fn(key, unused_t):
            next_key, init_env_key, rollout_key = jax.random.split(key, 3)
            env_state = self.env_reset(jax.random.split(init_env_key, num=pop_size))

            if self.discount == 1.0:
                # use fast undiscount evaluation
                env_state, metrics = fast_eval_rollout_episode(
                    self.env_step,
                    self.agent_eval_actions,
                    env_state,
                    pop_agent_state,
                    jax.random.split(rollout_key, num=pop_size),
                    self.max_episode_steps,
                    metric_names=metric_names,
                )

            else:
                env_state, episode_trajectory = eval_rollout_episode(
                    self.env_step,
                    self.agent_eval_actions,
                    env_state,
                    pop_agent_state,
                    jax.random.split(rollout_key, num=pop_size),
                    self.max_episode_steps,
                    metric_names=self.metric_names,
                )

                metrics = PyTreeDict()
                for name in metric_names:
                    if "reward" in name:
                        # For metrics like 'reward_forward' and 'reward_ctrl'
                        metrics[name] = compute_discount_return(
                            episode_trajectory.rewards[name],
                            episode_trajectory.dones,
                            self.discount,
                        )
                    elif "episode_length" == name:
                        metrics[name] = compute_episode_length(episode_trajectory.dones)
                    else:
                        # For other metrics like 'x_position', we use the last value as the objective.
                        # Note: It is ok to use [-1], since wrapper ensures that the last value
                        # repeats the terminal step value.
                        metrics[name] = episode_trajectory.rewards[name][-1]

            objectives = PyTreeDict(
                {
                    name: val
                    for name, val in metrics.items()
                    if name in self.metric_names
                }
            )
            sampled_timesteps = metrics.episode_length

            return next_key, (objectives, sampled_timesteps)  # [#envs]

        # [#iters, #pop, #envs]
        key, (objectives, sampled_timesteps) = jax.lax.scan(
            _evaluate_fn, state.key, (), length=self.num_iters
        )

        for k in objectives.keys():
            objective = jax.lax.collapse(
                jnp.swapaxes(objectives[k], 0, 1), 1, 3
            )  # [#pop, num_episodes]
            # by default, we use the mean value over different episodes.
            objectives[k] = self.reduce_fn[k](objective, axis=-1)

        if self.flatten_objectives:
            objectives = jnp.stack(list(objectives.values()), axis=-1)

        sampled_episodes = pop_size * self.num_episodes * self.env.num_envs
        sampled_timesteps = sampled_timesteps.sum()

        state = state.replace(
            key=key,
            sampled_timesteps=sampled_timesteps,
            sampled_episodes=sampled_episodes,
        )

        return objectives, state


def eval_env_step(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[
        [AgentState, SampleBatch, chex.PRNGKey], tuple[Action, PolicyExtraInfo]
    ],
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    sample_batch: SampleBatch,
    key: chex.PRNGKey,
    metric_names: tuple[str] = (),
) -> tuple[EnvState, SampleBatch]:
    actions, policy_extras = action_fn(agent_state, sample_batch, key)
    env_nstate = env_fn(env_state, actions)

    # info = env_nstate.info
    # env_extras = {x: info[x] for x in env_extra_fields if x in info}

    rewards = PyTreeDict(
        {
            name: val
            for name, val in env_nstate.info.metrics.items()
            if name in metric_names
        }
    )
    rewards.reward = env_nstate.reward

    transition = SampleBatch(
        rewards=rewards,
        dones=env_nstate.done,
    )

    return env_nstate, transition


def eval_rollout_episode(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[
        [AgentState, SampleBatch, chex.PRNGKey], tuple[Action, PolicyExtraInfo]
    ],
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    metric_names: tuple[str] = (),
) -> tuple[EnvState, SampleBatch]:
    """
    Collect given rollout_length trajectory.
    Avoid unnecessary env_step()

    Args:
        env: vmapped env w/o autoreset
    """

    _eval_env_step = partial(
        eval_env_step, env_fn, action_fn, metric_names=metric_names
    )

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
            env_state,
            agent_state,
            sample_batch,
            current_key,
        )

        return (env_nstate, next_key, transition), transition

    # run one-step rollout first to get bootstrap transition
    # it will not include in the trajectory when env_state is from env.reset()
    # this is manually controlled by user.
    _, transition = _eval_env_step(
        env_state, agent_state, SampleBatch(obs=env_state.obs), key
    )

    (env_state, _, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key, transition), (), length=rollout_length
    )

    return env_state, trajectory


def fast_eval_rollout_episode(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[
        [AgentState, SampleBatch, chex.PRNGKey], tuple[Action, PolicyExtraInfo]
    ],
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    metric_names: tuple[str] = (),
) -> tuple[EnvState, SampleBatch]:
    """
    Collect given rollout_length trajectory.
    Avoid unnecessary env_step()
    Args:
        env: vmapped env w/o autoreset
    """

    _eval_env_step = partial(
        eval_env_step, env_fn, action_fn, metric_names=metric_names
    )

    def _terminate_cond(carry):
        env_state, current_key, prev_metrics = carry
        return (prev_metrics.episode_length < rollout_length).all() & (
            ~env_state.done.all()
        )

    def _one_step_rollout(carry):
        """
        sample_batch: one-step obs
        transition: one-step full info
        """
        env_state, current_key, prev_metrics = carry
        next_key, current_key = rng_split(current_key, 2)

        sample_batch = SampleBatch(
            obs=env_state.obs,
        )

        transition, env_nstate = _eval_env_step(
            env_state, agent_state, sample_batch, current_key
        )

        metrics = PyTreeDict()
        for name in metric_names:
            if "reward" in name:
                metrics[name] = (
                    prev_metrics[name]
                    + (1 - transition.dones) * transition.rewards[name]
                )
            elif "episode_length" == name:
                metrics[name] = prev_metrics[name] + (1 - transition.dones)
            elif name in metrics:
                metrics[name] = transition.rewards[name]

        return env_nstate, next_key, metrics

    batch_shape = env_state.reward.shape

    env_state, _, metrics = jax.lax.while_loop(
        _terminate_cond,
        _one_step_rollout,
        (
            env_state,
            key,
            PyTreeDict(
                {
                    name: jnp.zeros(batch_shape, dtype=jnp.float32)
                    for name in metric_names
                }
            ),
        ),
    )

    return env_state, metrics
