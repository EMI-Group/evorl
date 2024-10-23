import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import PyTreeDict
from evorl.utils.jax_utils import right_shift_with_padding, tree_stop_gradient
from evorl.utils.rl_toolkits import (
    flatten_rollout_trajectory,
    flatten_pop_rollout_episode,
)

from ..td3 import TD3TrainMetric


DUMMY_TD3_TRAINMETRIC = TD3TrainMetric(
    critic_loss=jnp.zeros(()),
    actor_loss=jnp.zeros(()),
    raw_loss_dict=PyTreeDict(
        critic_loss=jnp.zeros(()),
        q_value=jnp.zeros(()),
        actor_loss=jnp.zeros(()),
    ),
)


def create_dummy_td3_trainmetric(num: int) -> TD3TrainMetric:
    if num >= 1:
        return DUMMY_TD3_TRAINMETRIC.replace(
            raw_loss_dict=jtu.tree_map(
                lambda x: jnp.broadcast_to(x, (num, *x.shape)),
                DUMMY_TD3_TRAINMETRIC.raw_loss_dict,
            )
        )
    else:
        raise ValueError(f"num should be positive, got {num}")


def rollout_episode(
    agent_state,
    replay_buffer_state,
    key,
    *,
    collector,
    replay_buffer,
    agent_state_vmap_axes,
    num_episodes,
    num_agents,
):
    chex.assert_tree_shape_prefix(agent_state, (num_agents,))

    eval_metrics, trajectory = jax.vmap(
        collector.rollout,
        in_axes=(agent_state_vmap_axes, 0, None),
    )(
        agent_state,
        jax.random.split(key, num_agents),
        num_episodes,
    )

    # [n, T, B, ...] -> [T, n*B, ...]
    trajectory = trajectory.replace(next_obs=None)
    trajectory = flatten_pop_rollout_episode(trajectory)

    mask = jnp.logical_not(right_shift_with_padding(trajectory.dones, 1))
    trajectory = trajectory.replace(dones=None)
    trajectory, mask = tree_stop_gradient(
        flatten_rollout_trajectory((trajectory, mask))
    )
    replay_buffer_state = replay_buffer.add(replay_buffer_state, trajectory, mask)

    return eval_metrics, trajectory, replay_buffer_state
