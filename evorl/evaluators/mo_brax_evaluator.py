import copy
import logging
from collections.abc import Callable
from functools import partial
import math

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from evorl.agent import AgentState, AgentActionFn
from evorl.envs import EnvState, EnvStepFn
from evorl.envs.brax import BraxAdapter
from evorl.rollout import SampleBatch
from evorl.types import Action, PolicyExtraInfo, PyTreeDict, pytree_field
from evorl.utils.jax_utils import rng_split
from evorl.utils.rl_toolkits import compute_discount_return, compute_episode_length

from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class BraxEvaluator(Evaluator):
    """Mutli-objective evaluator for Brax environments.

    Attributes:
        metric_names: The names of the metrics to evaluate, default is ("reward", "episode_lengths")

    """

    metric_names: tuple[str] = pytree_field(
        default=("reward", "episode_lengths"), static=True
    )

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.env.unwrapped, BraxAdapter), (
            "only support Brax environments"
        )

    def evaluate(
        self, agent_state: chex.ArrayTree, key: chex.PRNGKey, num_episodes: int
    ) -> chex.ArrayTree:
        num_envs = self.env.num_envs
        num_iters = math.ceil(num_episodes / num_envs)
        if num_episodes % num_envs != 0:
            logger.warning(
                f"num_episode ({num_episodes}) cannot be divided by parallel_envs ({num_envs}),"
                f"set new num_episodes={num_iters * num_envs}"
            )

        action_fn = self.action_fn
        env_reset_fn = self.env.reset
        env_step_fn = self.env.step
        if key.ndim > 1:
            for _ in range(key.ndim - 1):
                action_fn = jax.vmap(action_fn)
                env_reset_fn = jax.vmap(env_reset_fn)
                env_step_fn = jax.vmap(env_step_fn)

        metric_names = copy.deepcopy(tuple(self.metric_names))
        # we also need episode_length to calculate the sampled_timesteps
        if "episode_lengths" not in metric_names:
            metric_names = metric_names + ("episode_lengths",)

        def _evaluate_fn(key, unused_t):
            next_key, init_env_key, rollout_key = rng_split(key, 3)
            env_state = env_reset_fn(init_env_key)

            if self.discount == 1.0:
                # use fast undiscount evaluation
                metrics, env_state = fast_eval_metrics(
                    env_step_fn,
                    action_fn,
                    env_state,
                    agent_state,
                    rollout_key,
                    self.max_episode_steps,
                    metric_names=metric_names,
                )

            else:
                metrics, env_state = eval_metrics(
                    env_step_fn,
                    action_fn,
                    env_state,
                    agent_state,
                    rollout_key,
                    self.max_episode_steps,
                    discount=self.discount,
                    metric_names=self.metric_names,
                )

            return next_key, metrics  # [..., #envs]

        # [#iters, #pop, #envs]
        _, objectives = jax.lax.scan(_evaluate_fn, key, (), length=num_iters)

        objectives = jtu.tree_map(_flatten_metric, objectives)  # [#pop, num_episodes]

        return objectives


def _flatten_metric(x):
    """Flatten the last two dims.

    Args:
        x: jax tensor with shape (#iters, ..., #envs)

    Returns:
        flatten x with shape (..., #iters * #envs)
    """
    return jax.lax.collapse(jnp.moveaxis(x, 0, -2), -2)


def eval_env_step(
    env_fn: EnvStepFn,
    action_fn: AgentActionFn,
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    key: chex.PRNGKey,
    metric_names: tuple[str] = (),
) -> tuple[SampleBatch, EnvState]:
    # sample_batch: [#envs, ...]
    sample_batch = SampleBatch(obs=env_state.obs)

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

    return transition, env_nstate


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
) -> tuple[SampleBatch, EnvState]:
    """Evaulate a batch of episodic trajectories.

    The retruned metrics are defined by `metric_names`.
    """
    _eval_env_step = partial(
        eval_env_step, env_fn, action_fn, metric_names=metric_names
    )

    def _one_step_rollout(carry, unused_t):
        env_state, current_key, prev_transition = carry
        # next_key, current_key = jax.random.split(current_key, 2)
        next_key, current_key = rng_split(current_key, 2)

        transition, env_nstate = jax.lax.cond(
            env_state.done.all(),
            lambda *x: (env_state.replace(), prev_transition.replace()),
            _eval_env_step,
            env_state,
            agent_state,
            current_key,
        )

        return (env_nstate, next_key, transition), transition

    # run one-step rollout first to get bootstrap transition
    # it will not include in the trajectory when env_state is from env.reset()
    # this is manually controlled by user.
    bootstrap_transition, _ = _eval_env_step(env_state, agent_state, key)

    (env_state, _, _), trajectory = jax.lax.scan(
        _one_step_rollout,
        (env_state, key, bootstrap_transition),
        (),
        length=rollout_length,
    )

    return trajectory, env_state


def eval_metrics(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[
        [AgentState, SampleBatch, chex.PRNGKey], tuple[Action, PolicyExtraInfo]
    ],
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    discount: float,
    metric_names: tuple[str] = (),
) -> tuple[PyTreeDict, EnvState]:
    episode_trajectory, env_state = eval_rollout_episode(
        env_fn,
        action_fn,
        env_state,
        agent_state,
        key,
        rollout_length,
        metric_names=metric_names,
    )

    metrics = PyTreeDict()
    for name in metric_names:
        if "reward" in name:
            # For metrics like 'reward_forward' and 'reward_ctrl'
            metrics[name] = compute_discount_return(
                episode_trajectory.rewards[name],
                episode_trajectory.dones,
                discount,
            )
        elif "episode_lengths" == name:
            metrics[name] = compute_episode_length(episode_trajectory.dones)
        else:
            # For other metrics like 'x_position', we use the last value as the objective.
            # Note: It is ok to use [-1], since wrapper ensures that the last value
            # repeats the terminal step value.
            metrics[name] = episode_trajectory.rewards[name][-1]

    return metrics, env_state


def fast_eval_metrics(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[
        [AgentState, SampleBatch, chex.PRNGKey], tuple[Action, PolicyExtraInfo]
    ],
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    metric_names: tuple[str] = (),
) -> tuple[PyTreeDict, EnvState]:
    """Fast evaulate a batch of episodic trajectories.

    The retruned metrics are defined by `metric_names`.
    """
    _eval_env_step = partial(
        eval_env_step, env_fn, action_fn, metric_names=metric_names
    )

    def _terminate_cond(carry):
        env_state, current_key, prev_metrics = carry
        return (prev_metrics.episode_lengths < rollout_length).all() & (
            ~env_state.done.all()
        )

    def _one_step_rollout(carry):
        env_state, current_key, prev_metrics = carry
        next_key, current_key = rng_split(current_key, 2)

        transition, env_nstate = _eval_env_step(env_state, agent_state, current_key)

        prev_dones = env_state.done

        metrics = PyTreeDict()
        for name in metric_names:
            if "reward" in name:
                metrics[name] = (
                    prev_metrics[name] + (1 - prev_dones) * transition.rewards[name]
                )
            elif "episode_lengths" == name:
                metrics[name] = prev_metrics[name] + (1 - prev_dones)
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
            PyTreeDict({name: jnp.zeros(batch_shape) for name in metric_names}),
        ),
    )

    return metrics, env_state
