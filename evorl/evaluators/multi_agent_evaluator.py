"""Episodic evaluation for simultaneous-agent environments."""

import math

import jax
import jax.numpy as jnp

from evorl.distributed.sharding import mark_varying
from evorl.metrics import EvaluateMetric
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict, PyTreeNode, pytree_field


class MultiAgentEvaluator(PyTreeNode):
    """Evaluate per-agent returns with a shared episode boundary.

    Rewards remain keyed by agent, with scalar or vector values. Returns keep
    an episode axis followed by any objective axis. Each agent stops accumulating
    reward after its own termination; the episode length follows `done['__all__']`.
    Evaluation environments must use DISABLED autoreset.
    """

    env: object = pytree_field(static=True)
    action_fn: object = pytree_field(static=True)
    max_episode_steps: int = pytree_field(static=True)
    discount: float = pytree_field(default=1.0, static=True)

    def evaluate(self, agent_state, key, num_episodes):
        def episode(key, _):
            key, reset_key, rollout_key = jax.random.split(key, 3)
            state = self.env.reset(reset_key)
            axes = jax.sharding.get_abstract_mesh().manual_axes
            returns = PyTreeDict(
                {name: jnp.zeros_like(state.reward[name]) for name in self.env.agents}
            )
            lengths = jnp.zeros_like(state.done["__all__"], dtype=jnp.int32)

            def step(carry, t):
                state, key, returns, lengths = carry
                key, action_key = jax.random.split(key)
                actions, _ = self.action_fn(
                    agent_state, SampleBatch(obs=state.obs), action_key
                )
                next_state = self.env.step(state, actions)
                active_episode = ~state.done["__all__"].astype(jnp.bool_)
                next_returns = PyTreeDict()
                for name in self.env.agents:
                    reward = next_state.reward[name]
                    active = active_episode & ~state.done[name].astype(jnp.bool_)
                    if reward.ndim == active.ndim + 1:
                        active = active[..., None]
                    next_returns[name] = returns[name] + self.discount**t * jnp.where(
                        active, reward, 0
                    )
                return (
                    next_state,
                    key,
                    next_returns,
                    lengths + active_episode.astype(jnp.int32),
                ), None

            (_, _, returns, lengths), _ = jax.lax.scan(
                step,
                mark_varying((state, rollout_key, returns, lengths), axes),
                jnp.arange(self.max_episode_steps),
            )
            return key, (returns, lengths)

        _, (returns, lengths) = jax.lax.scan(
            episode, key, (), length=math.ceil(num_episodes / self.env.num_envs)
        )
        return EvaluateMetric(
            episode_return=jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), returns),
            episode_length=lengths.reshape(-1),
        )
