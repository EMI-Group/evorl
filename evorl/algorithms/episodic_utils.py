"""Shared episodic collection, replay setup and TD3 actor-state utilities."""

from flax import nnx
import jax
import jax.numpy as jnp

from evorl.agent import RandomAgent
from evorl.algorithms.offpolicy_utils import clean_trajectory
from evorl.algorithms.td3 import TD3TrainMetric
from evorl.networks.utils import model_state_axes
from evorl.rollout import rollout
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils import running_statistics
from evorl.utils.jax_utils import right_shift_with_padding, tree_stop_gradient
from evorl.utils.rl_toolkits import (
    flatten_pop_rollout_episode,
    flatten_rollout_trajectory,
)


def init_replay_buffer(env, replay_buffer, key):
    """Initialize replay storage with the episodic collector's transition layout."""
    obs = env.obs_space.sample(key)
    sample = SampleBatch(
        obs=obs,
        actions=jnp.zeros(env.action_space.shape),
        rewards=jnp.zeros(()),
        extras=PyTreeDict(
            policy_extras=PyTreeDict(),
            env_extras=PyTreeDict(ori_obs=obs, termination=jnp.zeros(())),
        ),
    )
    return replay_buffer.init(sample)


def prefill_replay_buffer(
    env, replay_buffer, agent_state, buffer_state, key, timesteps
):
    """Collect random transitions and initialize observation statistics.

    The caller supplies a separate NORMAL-autoreset environment. Preserve the
    episodic workflows' existing convention of excluding prefill from counters.
    """
    key, reset_key, rollout_key = jax.random.split(key, 3)
    random_agent = RandomAgent()
    random_state = random_agent.init(env.obs_space, env.action_space, jax.random.key(0))
    length = timesteps // env.num_envs
    trajectory, _ = rollout(
        env.step,
        random_agent.compute_actions,
        env.reset(reset_key),
        random_state,
        rollout_key,
        rollout_length=length,
        env_extra_fields=("ori_obs", "termination"),
    )
    trajectory = tree_stop_gradient(
        flatten_rollout_trajectory(clean_trajectory(trajectory))
    )
    buffer_state = replay_buffer.add(buffer_state, trajectory)
    if agent_state.obs_preprocessor_state is not None and length > 0:
        agent_state = agent_state.replace(
            obs_preprocessor_state=running_statistics.update(
                agent_state.obs_preprocessor_state, trajectory.obs
            )
        )
    return agent_state, buffer_state, key


def collect_population_episodes(
    agent_state,
    buffer_state,
    key,
    *,
    collector,
    replay_buffer,
    agent_state_vmap_axes,
    num_episodes,
    num_agents,
):
    """Collect population episodes and store only transitions before padding."""
    if isinstance(agent_state_vmap_axes.model_state, nnx.StateAxes):
        agent_state_vmap_axes = agent_state_vmap_axes.replace(
            model_state=model_state_axes(
                agent_state.model_state, agent_state_vmap_axes.model_state
            )
        )
    metrics, trajectory = jax.vmap(
        collector.rollout, in_axes=(agent_state_vmap_axes, 0, None)
    )(agent_state, jax.random.split(key, num_agents), num_episodes)
    trajectory = flatten_pop_rollout_episode(trajectory.replace(next_obs=None))
    mask = jnp.logical_not(right_shift_with_padding(trajectory.dones, 1))
    trajectory, mask = tree_stop_gradient(
        flatten_rollout_trajectory((trajectory.replace(dones=None), mask))
    )
    return metrics, trajectory, replay_buffer.add(buffer_state, trajectory, mask)


def import_td3_actor_params(actor, params):
    """Replace a TD3 MLP actor's genes and resize its auxiliary population axes.

    Only parameters evolve. Retain one member's auxiliary baseline when changing
    population size, then broadcast it to the incoming population. The caller
    owns this working NNX module and decides whether to synchronize a target.
    """
    previous = nnx.state(actor, nnx.Param)
    kernel = next(
        value.get_value()
        for path, value in nnx.to_flat_state(previous)
        if path[-1] == "kernel"
    )
    batch_ndim = 0 if kernel is None else kernel.ndim - 2
    auxiliary = jax.tree.map(
        lambda x: x[(0,) * batch_ndim], nnx.state(actor, nnx.Not(nnx.Param))
    )
    if params is None:
        params = jax.tree.map(lambda _: None, previous)
    else:
        kernel = next(
            value.get_value()
            for path, value in nnx.to_flat_state(params)
            if path[-1] == "kernel"
        )
        auxiliary = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (*kernel.shape[:-2], *x.shape)), auxiliary
        )
    nnx.update(actor, params, auxiliary)


def create_dummy_td3_trainmetric(num: int) -> TD3TrainMetric:
    """Create zero training metrics for an iteration without TD3 updates."""
    if num < 1:
        raise ValueError(f"num must be positive, got {num}")
    return TD3TrainMetric(
        critic_loss=jnp.zeros(()),
        actor_loss=jnp.zeros(()),
        raw_loss_dict=PyTreeDict(
            critic_loss=jnp.zeros(num),
            q_value=jnp.zeros(num),
            actor_loss=jnp.zeros(num),
        ),
    )
