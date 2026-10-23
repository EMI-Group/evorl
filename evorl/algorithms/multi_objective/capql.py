"""Concave-Augmented Pareto Q-Learning (Lu, Herman and Yu, ICLR 2023)."""

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig

from evorl.agent import Agent, AgentState
from evorl.distributed import pmean_nnx_loss_and_grad, psum
from evorl.distribution import get_tanh_norm_dist
from evorl.envs import AutoresetMode, Box, create_env
from evorl.evaluators import Evaluator
from evorl.metrics import TrainMetric
from evorl.networks import make_policy_network, make_multi_objective_q_network
from evorl.replay_buffers import ReplayBuffer
from evorl.types import PyTreeDict, pytree_field
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update

from .common import (
    MultiObjectiveOffPolicyWorkflow,
    objective_count,
    preference_inputs,
    validate_replay_config,
)


class CAPQLAgent(Agent):
    """Preference-conditioned stochastic actor and twin vector-valued critics.

    Entropy augments every objective equally. Since preferences sum to one,
    the scalarized entropy coefficient remains `alpha`, independent of the
    number of objectives. The pessimistic estimate takes the minimum between
    critics for each objective before scalarization, following the authors.
    """

    num_objectives: int = pytree_field(static=True)
    agent_network_cfg: DictConfig = pytree_field(static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)
    discount: float = 0.99
    alpha: float = 0.2

    def init(self, obs_space, action_space, key):
        if (
            not isinstance(action_space, Box)
            or len(action_space.shape) != 1
            or not jnp.issubdtype(action_space.low.dtype, jnp.floating)
        ):
            raise TypeError(
                "CAPQL requires a one-dimensional continuous Box action space."
            )
        cfg = self.agent_network_cfg
        if cfg.num_critics < 2:
            raise ValueError("CAPQL requires at least two critics.")
        actor_key, critic_key = jax.random.split(key)
        actor = make_policy_network(
            input_size=obs_space.shape[0] + self.num_objectives,
            action_size=action_space.shape[0] * 2,
            hidden_layer_sizes=cfg.actor_hidden_layer_sizes,
            norm_layer_type=cfg.get("norm_layer_type", "none"),
            rngs=nnx.Rngs(actor_key),
        )
        critic = make_multi_objective_q_network(
            input_size=obs_space.shape[0] + self.num_objectives,
            action_size=action_space.shape[0],
            n_stack=cfg.num_critics,
            num_objectives=self.num_objectives,
            hidden_layer_sizes=cfg.critic_hidden_layer_sizes,
            norm_layer_type=cfg.get("norm_layer_type", "none"),
            rngs=nnx.Rngs(critic_key),
        )
        return AgentState(
            model_state=nnx.unpack(
                nnx.Dict(actor=actor, critic=critic, target_critic=nnx.clone(critic))
            ),
            obs_preprocessor_state=running_statistics.init_state(
                jnp.zeros(obs_space.shape, jnp.float32)
            )
            if self.obs_preprocessor is not None
            else None,
            extra_state=PyTreeDict(
                preferences=jnp.full(
                    (self.num_objectives,), 1 / self.num_objectives, jnp.float32
                ),
                action_scale=(action_space.high - action_space.low) / 2,
                action_bias=(action_space.high + action_space.low) / 2,
            ),
        )

    def compute_actions(self, agent_state, sample_batch, key):
        inputs, _ = preference_inputs(self, agent_state, sample_batch)
        raw = self.get_models(agent_state, copy=True).actor(inputs)
        dist = get_tanh_norm_dist(*jnp.split(raw, 2, axis=-1))
        action = dist.sample(seed=key)
        return (
            action * agent_state.extra_state.action_scale
            + agent_state.extra_state.action_bias,
            PyTreeDict(),
        )

    def evaluate_actions(self, agent_state, sample_batch, key):
        inputs, _ = preference_inputs(self, agent_state, sample_batch)
        raw = self.get_models(agent_state, copy=True).actor(inputs)
        dist = get_tanh_norm_dist(*jnp.split(raw, 2, axis=-1))
        return (
            dist.mode() * agent_state.extra_state.action_scale
            + agent_state.extra_state.action_bias,
            PyTreeDict(),
        )

    def critic_loss(self, agent_state, sample_batch, key):
        models = self.get_models(agent_state)
        inputs, _ = preference_inputs(self, agent_state, sample_batch)
        next_inputs, _ = preference_inputs(
            self, agent_state, sample_batch.replace(obs=sample_batch.next_obs)
        )
        raw = models.actor(next_inputs)
        dist = get_tanh_norm_dist(*jnp.split(raw, 2, axis=-1))
        normalized = dist.sample(seed=key)
        scale, bias = (
            agent_state.extra_state.action_scale,
            agent_state.extra_state.action_bias,
        )
        next_actions = normalized * scale + bias
        logp = dist.log_prob(normalized) - jnp.log(scale).sum()
        # (..., critics, objectives): minimum is over critics, not objectives.
        next_q = models.target_critic(next_inputs, next_actions).min(axis=-2)
        target = jax.lax.stop_gradient(
            sample_batch.rewards
            + self.discount
            * (1 - sample_batch.extras.terminations[..., None])
            * (next_q - self.alpha * logp[..., None])
        )
        q = models.critic(inputs, sample_batch.actions)
        target = jnp.broadcast_to(target[..., None, :], q.shape)
        return PyTreeDict(
            critic_loss=optax.squared_error(q, target).sum(axis=-2).mean(),
            q_value=q.mean(),
        )

    def actor_loss(self, agent_state, sample_batch, key):
        models = self.get_models(agent_state)
        inputs, weights = preference_inputs(self, agent_state, sample_batch)
        raw = models.actor(inputs)
        dist = get_tanh_norm_dist(*jnp.split(raw, 2, axis=-1))
        normalized = dist.sample(seed=key)
        scale, bias = (
            agent_state.extra_state.action_scale,
            agent_state.extra_state.action_bias,
        )
        actions = normalized * scale + bias
        logp = dist.log_prob(normalized) - jnp.log(scale).sum()
        q = models.critic(inputs, actions).min(axis=-2)
        return PyTreeDict(
            actor_loss=(self.alpha * logp - (q * weights).sum(-1)).mean(),
            entropy=-logp.mean(),
        )


class CAPQLWorkflow(MultiObjectiveOffPolicyWorkflow):
    """Optimize the authors' entropy-augmented vector Bellman objective."""

    resample_preferences_every_step = True

    @classmethod
    def name(cls):
        return "CAPQL"

    @classmethod
    def _build_from_config(cls, config):
        validate_replay_config(config)
        if config.alpha < 0 or config.target_network_update_freq <= 0:
            raise ValueError(
                "alpha must be nonnegative and target update frequency positive."
            )
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )
        space = env.action_space
        if (
            not isinstance(space, Box)
            or len(space.shape) != 1
            or not jnp.issubdtype(space.low.dtype, jnp.floating)
        ):
            raise TypeError(
                "CAPQL requires a one-dimensional continuous Box action space."
            )
        if not bool(
            jnp.all(
                jnp.isfinite(space.low)
                & jnp.isfinite(space.high)
                & (space.high > space.low)
            )
        ):
            raise ValueError("CAPQL requires finite, strictly ordered action bounds.")
        agent = CAPQLAgent(
            num_objectives=objective_count(env),
            agent_network_cfg=config.agent_network,
            discount=config.discount,
            alpha=config.alpha,
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
        return state, PyTreeDict(
            actor=nnx.unpack(
                nnx.Optimizer(models.actor, self.optimizer, wrt=nnx.Param)
            ),
            critic=nnx.unpack(
                nnx.Optimizer(models.critic, self.optimizer, wrt=nnx.Param)
            ),
        )

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
        trajectory = tree_stop_gradient(flatten_rollout_trajectory(trajectory))
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
            key, sample_key, critic_key, actor_key = jax.random.split(key, 4)
            batch = self.replay_buffer.sample(buffer, sample_key)
            models = nnx.merge(agent_state.model_state, copy=True)

            def critic_loss(models):
                info = self.agent.critic_loss(
                    agent_state.replace(model_state=nnx.unpack(models)),
                    batch,
                    critic_key,
                )
                return info.critic_loss, info

            (critic_value, critic_info), grads = pmean_nnx_loss_and_grad(
                critic_loss,
                self.dp_axis_name,
                has_aux=True,
                wrt=nnx.All(nnx.Param, nnx.PathContains("critic")),
            )(models)
            critic_optimizer = nnx.merge(opt_state.critic, copy=True)
            critic_optimizer.update(models.critic, grads["critic"])
            auxiliary = nnx.clone(nnx.state(models.critic, nnx.Not(nnx.Param)))

            def actor_loss(models):
                info = self.agent.actor_loss(
                    agent_state.replace(model_state=nnx.unpack(models)),
                    batch,
                    actor_key,
                )
                return info.actor_loss, info

            (actor_value, actor_info), grads = pmean_nnx_loss_and_grad(
                actor_loss,
                self.dp_axis_name,
                has_aux=True,
                wrt=nnx.All(nnx.Param, nnx.PathContains("actor")),
            )(models)
            actor_optimizer = nnx.merge(opt_state.actor, copy=True)
            actor_optimizer.update(models.actor, grads["actor"])
            nnx.update(models.critic, auxiliary)
            count += 1

            def update_target(target):
                return soft_target_update(
                    target, nnx.state(models.critic, nnx.Param), self.config.tau
                )

            nnx.update(
                models.target_critic,
                jax.lax.cond(
                    count % self.config.target_network_update_freq == 0,
                    update_target,
                    lambda target: target,
                    nnx.state(models.target_critic, nnx.Param),
                ),
            )
            info = critic_info.replace(**actor_info)
            return (
                agent_state.replace(model_state=nnx.unpack(models)),
                PyTreeDict(
                    actor=nnx.unpack(actor_optimizer),
                    critic=nnx.unpack(critic_optimizer),
                ),
                key,
                count,
            ), (critic_value + actor_value, info)

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
