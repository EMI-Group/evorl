import jax
import jax.numpy as jnp
import chex

from typing import Dict
from evorl.types import Done


def sort_dict(d: Dict) -> Dict:
    return dict(sorted(d.items()))


def batchify(x: dict, agent_list, num_actors):
    max_dim = max([x[a].shape[-1] for a in agent_list])

    def pad(z, length):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + [length - z.shape[-1]])], -1)

    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a])
                  for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def multi_agent_episode_done(done: Done) -> chex.Array:
    return done['__all__']
