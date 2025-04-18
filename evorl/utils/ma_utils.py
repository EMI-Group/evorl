import chex
import jax
import jax.numpy as jnp

from evorl.types import AgentID, Done


def batchify(x: dict[AgentID, jax.Array], agent_list, padding=False) -> jax.Array:
    """Batchify the data for multi-agent training.

    Output batched data.

    Args:
        x: data from each agent, [batch_dims..., val]
            Note: Currently, only the last dimension is viewed as value, and the rest are batch dimensions.
        agent_list: list, list of agent names
        num_actors: int, number of actors
        padding: bool, whether to pad the data to the same length over the last dimension.
            set to False if the data already has same length.

    Returns:
        Concatenated data from multiple agents with shape: [num_actors, batch_dims..., val]
    """
    if padding:

        def _pad(z, length):
            return jnp.concatenate(
                [z, jnp.zeros(z.shape[:-1] + [length - z.shape[-1]])], -1
            )

        max_dim = max([x[a].shape[-1] for a in agent_list])
        x = jnp.stack(
            [
                x[a] if x[a].shape[-1] == max_dim else _pad(x[a], max_dim)
                for a in agent_list
            ]
        )
    else:
        x = jnp.stack([x[a] for a in agent_list])

    return x  # [num_actors, batch_dims..., val]


def unbatchify(x: jax.Array, agent_list) -> dict[AgentID, jax.Array]:
    """Unbatchify the data for multi-agent training.

    Here we assume data like actions has the same shape for each agent. (True for MaBrax)

    Args:
        x: batched data, [num_actors, batch_dims..., val]
            Note: Currently, only the last dimension is viewed as value, and the rest are batch dimensions.
        agent_list: list, list of agent names

    Returns:
        Dict {agent_name: data}
    """
    return {a: x[i] for i, a in enumerate(agent_list)}


def multi_agent_episode_done(done: Done) -> chex.Array:
    """Check whether the multi-agent episode is done."""
    return done["__all__"]
