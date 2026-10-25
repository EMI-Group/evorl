"""Joint replay collection shared by continuous-action MARL workflows."""

import math
from functools import partial

from flax import nnx
import jax
import jax.numpy as jnp

from evorl.algorithms.offpolicy_utils import OffPolicyWorkflowTemplate
from evorl.distributed import psum
from evorl.metrics import EvaluateMetric
from evorl.sample_batch import SampleBatch
from evorl.rollout import multi_agent_rollout
from evorl.types import PyTreeDict
from evorl.utils.jax_utils import tree_stop_gradient, rng_fold_in_axis
from evorl.utils.rl_toolkits import flatten_rollout_trajectory

from .common import scale_action, update_preprocessing


def random_actions(agent, agent_state, batch, key):
    """Sample bounded actions for the shared joint replay warmup."""
    actions = PyTreeDict()
    keys = jax.random.split(key, len(agent.agents))
    for index, name in enumerate(agent.agents):
        # All leaves have the same environment batch prefix during collection.
        value = jax.tree.leaves(batch.obs[name])[0]
        shape = (*value.shape[:1], agent.action_sizes[index])
        normalized = jax.random.uniform(keys[index], shape, minval=-1, maxval=1)
        actions[name] = scale_action(
            normalized, agent.action_bounds[index], agent.action_shapes[index]
        )
    return actions, PyTreeDict()


def joint_actions(agent, actions):
    """Concatenate agent actions in their fixed environment order."""
    parts = []
    for name, shape in zip(agent.agents, agent.action_shapes, strict=True):
        value = actions[name]
        batch_shape = value.shape[: -len(shape)] if shape else value.shape
        parts.append(value.reshape((*batch_shape, -1)))
    return jnp.concatenate(parts, axis=-1)


class MultiAgentOffPolicyWorkflow(OffPolicyWorkflowTemplate):
    """Preserve aligned joint samples and per-agent training/optimizer state."""

    def _setup_agent_and_optimizer(self, key):
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        models = nnx.merge(agent_state.model_state, copy=True)
        opt_state = PyTreeDict(
            {
                name: PyTreeDict(
                    actor=nnx.unpack(
                        nnx.Optimizer(models[name].actor, self.optimizer, wrt=nnx.Param)
                    ),
                    critic=nnx.unpack(
                        nnx.Optimizer(
                            models[name].critic, self.optimizer, wrt=nnx.Param
                        )
                    ),
                )
                for name in self.agent.agents
            }
        )
        return agent_state, opt_state

    def _setup_replaybuffer(self, key):
        abstract = jax.eval_shape(self.env.reset, key)
        unbatch = lambda tree: jax.tree.map(
            lambda x: jnp.zeros(x.shape[1:], x.dtype), tree
        )
        spec = SampleBatch(
            obs=unbatch(abstract.obs),
            next_obs=unbatch(abstract.obs),
            actions=PyTreeDict(
                {
                    name: space.sample(key)
                    for name, space in self.env.action_space.items()
                }
            ),
            rewards=PyTreeDict(
                {name: unbatch(abstract.reward[name]) for name in self.agent.agents}
            ),
            dones=PyTreeDict(
                {name: unbatch(abstract.done[name]) for name in self.agent.agents}
            ),
            extras=PyTreeDict(
                policy_extras=PyTreeDict(),
                active=PyTreeDict(
                    {name: jnp.bool_(True) for name in self.agent.agents}
                ),
                terminations=PyTreeDict(
                    {
                        name: unbatch(abstract.info.termination[name])
                        for name in self.agent.agents
                    }
                ),
                episode_done=unbatch(abstract.done["__all__"]),
                global_obs=unbatch(abstract.info.get("global_obs")),
                next_global_obs=unbatch(abstract.info.get("global_obs")),
            ),
        )
        return self.replay_buffer.init(spec)

    def _postsetup_replaybuffer(self, state):
        key, random_key, policy_key = jax.random.split(state.key, 3)
        random_key, policy_key = rng_fold_in_axis(
            (random_key, policy_key), self.dp_axis_name
        )
        random_length = self.config.random_timesteps // self.config.num_envs
        policy_length = max(
            0,
            math.ceil(
                (
                    self.config.learning_start_timesteps
                    - random_length * self.config.num_envs
                )
                / self.config.num_envs
            ),
        )
        agent_state, buffer_state, env_state = (
            state.agent_state,
            state.replay_buffer_state,
            state.env_state,
        )
        sampled_steps = jnp.uint32(0)
        for length, rollout_key, action_fn in (
            (random_length, random_key, partial(random_actions, self.agent)),
            (policy_length, policy_key, self.agent.compute_actions),
        ):
            if length == 0:
                continue
            reset_key, rollout_key = jax.random.split(rollout_key)
            trajectory, env_state = multi_agent_rollout(
                self.env,
                action_fn,
                self.env.reset(reset_key),
                agent_state,
                rollout_key,
                length,
            )
            sampled_steps += jnp.uint32(length * self.config.num_envs)
            trajectory = tree_stop_gradient(flatten_rollout_trajectory(trajectory))
            agent_state = update_preprocessing(
                self.agent, agent_state, trajectory, self.dp_axis_name
            )
            buffer_state = self.replay_buffer.add(buffer_state, trajectory)
        metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps
            + psum(sampled_steps, self.dp_axis_name),
        )
        if random_length + policy_length > 0:
            key, env_key = jax.random.split(key)
            env_key = rng_fold_in_axis(env_key, self.dp_axis_name)
            env_state = self.env.reset(env_key)
        return state.replace(
            key=key,
            metrics=metrics,
            env_state=env_state,
            agent_state=agent_state,
            replay_buffer_state=buffer_state,
        )

    def evaluate(self, state):
        key, eval_key = jax.random.split(state.key)
        eval_key = rng_fold_in_axis(eval_key, self.dp_axis_name)
        metrics = self.evaluator.evaluate(
            state.agent_state, eval_key, self.config.eval_episodes
        )
        return EvaluateMetric(
            episode_returns=jax.tree.map(jnp.mean, metrics.episode_returns),
            episode_lengths=metrics.episode_lengths.mean(),
        ).all_reduce(self.dp_axis_name), state.replace(key=key)
