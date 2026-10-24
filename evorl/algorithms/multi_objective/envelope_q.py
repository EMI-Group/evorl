"""Envelope Q-learning (Yang, Sun and Narasimhan, NeurIPS 2019)."""

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig

from evorl.agent import Agent, AgentState
from evorl.algorithms.offpolicy_utils import clean_trajectory
from evorl.distributed import pmean_nnx_loss_and_grad, psum
from evorl.envs import AutoresetMode, Discrete, create_env
from evorl.evaluators import Evaluator
from evorl.metrics import TrainMetric
from evorl.networks import make_multi_objective_discrete_q_network
from evorl.replay_buffers import ReplayBuffer
from evorl.types import PyTreeDict, pytree_field
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update

from .common import (
    MultiObjectiveOffPolicyWorkflow,
    objective_count,
    preference_inputs,
    sample_preferences,
    validate_replay_config,
)


def envelope_next_values(online_values, target_values, preferences):
    """Select an entire value vector by maximizing over actions AND preferences.

    Args:
        online_values: Next Q-values with shape `(batch, candidates, actions, objectives)`.
        target_values: Target-network values with the same shape.
        preferences: Query weights with shape `(queries, objectives)`.

    Returns:
        Target vectors with shape `(batch, queries, objectives)`. Double-Q
        selection uses the online network; all coordinates are gathered from
        the same selected target vector, rather than maximized independently.
    """
    scores = jnp.einsum("wm,bkam->bwka", preferences, online_values)
    indices = scores.reshape((*scores.shape[:2], -1)).argmax(-1)
    targets = target_values.reshape(
        (target_values.shape[0], -1, target_values.shape[-1])
    )
    return jnp.take_along_axis(targets, indices[..., None], axis=1)


class EnvelopeQAgent(Agent):
    """A preference-conditioned vector Q-function over discrete actions."""

    num_objectives: int = pytree_field(static=True)
    agent_network_cfg: DictConfig = pytree_field(static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)
    num_sample_preferences: int = pytree_field(default=4, static=True)
    discount: float = 0.99
    initial_epsilon: float = 0.8
    initial_homotopy_lambda: float = 0.0

    def init(self, obs_space, action_space, key):
        if not isinstance(action_space, Discrete):
            raise TypeError("EnvelopeQ requires scalar Discrete actions.")
        q = make_multi_objective_discrete_q_network(
            input_size=obs_space.shape[0] + self.num_objectives,
            action_size=action_space.n,
            hidden_layer_sizes=self.agent_network_cfg.q_hidden_layer_sizes,
            num_objectives=self.num_objectives,
            rngs=nnx.Rngs(key),
        )
        return AgentState(
            model_state=nnx.unpack(nnx.Dict(q=q, target_q=nnx.clone(q))),
            obs_preprocessor_state=running_statistics.init_state(
                jnp.zeros(obs_space.shape, jnp.float32)
            )
            if self.obs_preprocessor is not None
            else None,
            extra_state=PyTreeDict(
                preferences=jnp.full(
                    (self.num_objectives,), 1 / self.num_objectives, jnp.float32
                ),
                exploration_epsilon=jnp.float32(self.initial_epsilon),
                homotopy_lambda=jnp.float32(self.initial_homotopy_lambda),
            ),
        )

    def compute_actions(self, agent_state, sample_batch, key):
        inputs, weights = preference_inputs(self, agent_state, sample_batch)
        values = self.get_models(agent_state, copy=True).q(inputs)
        greedy = (values * weights[..., None, :]).sum(-1).argmax(-1)
        explore_key, action_key = jax.random.split(key)
        random = jax.random.randint(action_key, greedy.shape, 0, values.shape[-2])
        explore = (
            jax.random.uniform(explore_key, greedy.shape)
            < agent_state.extra_state.exploration_epsilon
        )
        return jnp.where(explore, random, greedy), PyTreeDict()

    def evaluate_actions(self, agent_state, sample_batch, key):
        inputs, weights = preference_inputs(self, agent_state, sample_batch)
        values = self.get_models(agent_state, copy=True).q(inputs)
        return (values * weights[..., None, :]).sum(-1).argmax(-1), PyTreeDict()

    def loss(self, agent_state, sample_batch, key):
        """Use hindsight preferences and the paper's Eq. 6/7 homotopy objective.

        The vector loss is squared L2 error; the auxiliary scalar loss is
        absolute projected error. The released Torch code instead squares the
        projected error; here the default follows the paper's stated equation.
        """
        models = self.get_models(agent_state)
        weights = sample_preferences(
            key, (self.num_sample_preferences,), self.num_objectives
        )
        batch_size = sample_batch.obs.shape[0]
        obs = jnp.broadcast_to(
            sample_batch.obs[:, None, :],
            (batch_size, self.num_sample_preferences, sample_batch.obs.shape[-1]),
        )
        next_obs = jnp.broadcast_to(
            sample_batch.extras.env_extras.ori_obs[:, None, :], obs.shape
        )
        relabeled = sample_batch.replace(
            obs=obs, extras=sample_batch.extras.replace(preferences=weights)
        )
        inputs, _ = preference_inputs(self, agent_state, relabeled)
        next_inputs, _ = preference_inputs(
            self, agent_state, relabeled.replace(obs=next_obs)
        )
        values = models.q(inputs)
        values = jnp.take_along_axis(
            values, sample_batch.actions[:, None, None, None], axis=-2
        ).squeeze(-2)
        next_values = envelope_next_values(
            models.q(next_inputs), models.target_q(next_inputs), weights
        )
        target = jax.lax.stop_gradient(
            sample_batch.rewards[:, None, :]
            + self.discount
            * (1 - sample_batch.extras.env_extras.termination[:, None, None])
            * next_values
        )
        error = values - target
        vector_loss = jnp.square(error).sum(-1).mean()
        scalar_loss = jnp.abs((error * weights).sum(-1)).mean()
        mix = agent_state.extra_state.homotopy_lambda
        return PyTreeDict(
            q_loss=(1 - mix) * vector_loss + mix * scalar_loss,
            vector_loss=vector_loss,
            scalar_loss=scalar_loss,
            q_value=values.mean(),
        )


class EnvelopeQWorkflow(MultiObjectiveOffPolicyWorkflow):
    """Learn one network for a set of linear reward preferences."""

    @classmethod
    def name(cls):
        return "EnvelopeQ"

    @classmethod
    def _build_from_config(cls, config):
        validate_replay_config(config)
        if config.num_sample_preferences <= 0 or config.target_network_update_freq <= 0:
            raise ValueError(
                "Preference samples and target update frequency must be positive."
            )
        if (
            min(
                config.exploration_epsilon.decay_updates,
                config.homotopy_lambda.decay_updates,
            )
            <= 0
        ):
            raise ValueError("Exploration and homotopy decay_updates must be positive.")
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )
        if not isinstance(env.action_space, Discrete):
            raise TypeError("EnvelopeQ requires scalar Discrete actions.")
        agent = EnvelopeQAgent(
            num_objectives=objective_count(env),
            agent_network_cfg=config.agent_network,
            discount=config.discount,
            num_sample_preferences=config.num_sample_preferences,
            initial_epsilon=config.exploration_epsilon.start,
            initial_homotopy_lambda=config.homotopy_lambda.start,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
        )
        optimizer = optax.adam(config.optimizer.lr)
        if config.optimizer.grad_clip_norm:
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.optimizer.grad_clip_norm), optimizer
            )
        replay = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            sample_batch_size=config.batch_size,
            min_sample_timesteps=config.batch_size,
        )
        evaluator = Evaluator(
            env=create_env(
                config.env,
                parallel=config.num_eval_envs,
                autoreset_mode=AutoresetMode.DISABLED,
            ),
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )
        return cls(env, agent, optimizer, evaluator, replay, config)

    def _setup_agent_and_optimizer(self, key):
        state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        models = self.agent.get_models(state, copy=True)
        return state, nnx.unpack(nnx.Optimizer(models.q, self.optimizer, wrt=nnx.Param))

    def step(self, state):
        key, rollout_key, learn_key = jax.random.split(state.key, 3)
        if self.is_multi_device:
            rank = jax.lax.axis_index(self.dp_axis_name)
            rollout_key, learn_key = (
                jax.random.fold_in(k, rank) for k in (rollout_key, learn_key)
            )
        trajectory, state = self._rollout(
            state, rollout_key, self.config.rollout_length
        )
        episodes = trajectory.dones.sum().astype(jnp.uint32)
        trajectory = tree_stop_gradient(
            flatten_rollout_trajectory(clean_trajectory(trajectory))
        )
        buffer = self.replay_buffer.add(state.replay_buffer_state, trajectory)
        agent_state = state.agent_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    axis_name=self.dp_axis_name,
                )
            )

        def update(carry, _):
            agent_state, opt_state, key, count = carry
            key, sample_key, loss_key = jax.random.split(key, 3)
            batch = self.replay_buffer.sample(buffer, sample_key)
            models = nnx.merge(agent_state.model_state, copy=True)

            def loss_fn(models):
                info = self.agent.loss(
                    agent_state.replace(model_state=nnx.unpack(models)), batch, loss_key
                )
                return info.q_loss, info

            (value, info), grads = pmean_nnx_loss_and_grad(
                loss_fn,
                self.dp_axis_name,
                has_aux=True,
                wrt=nnx.All(nnx.Param, nnx.PathContains("q")),
            )(models)
            optimizer = nnx.merge(opt_state, copy=True)
            optimizer.update(models.q, grads["q"])
            count += 1

            def update_target(target):
                return soft_target_update(
                    target, nnx.state(models.q, nnx.Param), self.config.tau
                )

            nnx.update(
                models.target_q,
                jax.lax.cond(
                    count % self.config.target_network_update_freq == 0,
                    update_target,
                    lambda target: target,
                    nnx.state(models.target_q, nnx.Param),
                ),
            )
            epsilon, mix = self.config.exploration_epsilon, self.config.homotopy_lambda
            agent_state = agent_state.replace(
                model_state=nnx.unpack(models),
                extra_state=agent_state.extra_state.replace(
                    exploration_epsilon=optax.linear_schedule(
                        epsilon.start, epsilon.end, epsilon.decay_updates
                    )(count),
                    homotopy_lambda=optax.linear_schedule(
                        mix.start, mix.end, mix.decay_updates
                    )(count),
                ),
            )
            return (agent_state, nnx.unpack(optimizer), key, count), (value, info)

        (agent_state, opt_state, _, updates), (loss, info) = jax.lax.scan(
            update,
            (agent_state, state.opt_state, learn_key, state.metrics.training_updates),
            (),
            length=self.config.num_updates_per_iter,
        )
        metrics = state.metrics.replace(
            iterations=state.metrics.iterations + 1,
            training_updates=updates,
            sampled_timesteps=state.metrics.sampled_timesteps
            + psum(
                jnp.uint32(self.config.num_envs * self.config.rollout_length),
                self.dp_axis_name,
            ),
            sampled_episodes=state.metrics.sampled_episodes
            + psum(episodes, self.dp_axis_name),
        )
        train = TrainMetric(
            loss=loss.mean(), raw_loss_dict=jax.tree.map(lambda x: x.mean(0), info)
        ).all_reduce(self.dp_axis_name)
        return train, state.replace(
            key=key,
            agent_state=agent_state,
            opt_state=opt_state,
            metrics=metrics,
            replay_buffer_state=buffer,
        )
