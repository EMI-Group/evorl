"""Fixed-preference multi-objective SAC for continuous and discrete actions.

MORL-Baselines' MOSAC scalarizes the vector critic before both the Bellman
regression and the pessimistic minimum. This differs from CAPQL's
preference-conditioned networks and coordinatewise vector Bellman loss.
"""

import math
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import DictConfig

from evorl.agent import Agent, AgentState, RandomAgent
from evorl.algorithms.offpolicy_utils import OffPolicyWorkflowTemplate, clean_trajectory
from evorl.distributed import DP_AXIS_NAME, pmean_nnx_loss_and_grad, psum
from evorl.distribution import get_tanh_norm_dist, get_categorical_dist
from evorl.envs import AutoresetMode, Box, Discrete, create_env
from evorl.evaluators import Evaluator
from evorl.metrics import TrainMetric
from evorl.networks import (
    make_policy_network,
    make_multi_objective_q_network,
    make_multi_objective_discrete_q_network,
)
from evorl.recorders import add_prefix
from evorl.replay_buffers import ReplayBuffer
from evorl.rollout import rollout
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict, State, pytree_field
from evorl.workflows.rl_workflow import _shard_map_replay_buffer_context
from evorl.utils import running_statistics
from evorl.utils.jax_utils import rng_fold_in_axis
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update

from .common import (
    MultiObjectiveEvaluateMetric,
    MultiObjectiveWorkflowMetric,
    objective_count,
    validate_replay_config,
)


def scalarized_q(values, preferences):
    """Project the final objective axis before comparing independent critics."""
    return jnp.sum(values * preferences, axis=-1)


class MOSACAgent(Agent):
    """An observation-only actor and twin vector critics for one fixed preference.

    `extra_state.preferences` is dynamic so MORL/D can change a subproblem's
    weights while sharing unscalarized replay. Critic coordinates are learned
    through their weighted projection, as in the reference implementation;
    individual coordinates are not claimed to estimate separate Bellman values.

    Discrete actor and temperature losses use `sum(action).mean(batch)`. The
    reference averages the action axis too, introducing an extra `1 / actions`
    factor; this implementation uses the categorical expectation directly.
    """

    num_objectives: int = pytree_field(static=True)
    continuous_action: bool = pytree_field(static=True)
    agent_network_cfg: DictConfig = pytree_field(static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)
    discount: float = 0.99
    init_alpha: float = 0.2
    target_entropy_ratio: float = 0.98

    def init(self, obs_space, action_space, key):
        cfg = self.agent_network_cfg
        if cfg.num_critics < 2:
            raise ValueError("MOSAC requires at least two critics")
        if self.continuous_action != isinstance(action_space, Box):
            raise TypeError("continuous_action must match the environment action space")
        key, critic_key, actor_key = jax.random.split(key, 3)
        action_size = (
            action_space.shape[0] if self.continuous_action else action_space.n
        )
        actor = make_policy_network(
            input_size=obs_space.shape[0],
            action_size=action_size * 2 if self.continuous_action else action_size,
            hidden_layer_sizes=cfg.actor_hidden_layer_sizes,
            rngs=nnx.Rngs(actor_key),
        )
        critic_factory = (
            make_multi_objective_q_network
            if self.continuous_action
            else make_multi_objective_discrete_q_network
        )
        critic = critic_factory(
            input_size=obs_space.shape[0],
            action_size=action_size,
            num_objectives=self.num_objectives,
            n_stack=cfg.num_critics,
            hidden_layer_sizes=cfg.critic_hidden_layer_sizes,
            rngs=nnx.Rngs(critic_key),
        )
        models = nnx.Dict(
            actor=actor,
            critic=critic,
            target_critic=nnx.clone(critic),
            log_alpha=nnx.Param(jnp.log(jnp.float32(self.init_alpha))),
        )
        entropy = (
            -float(action_size)
            if self.continuous_action
            else self.target_entropy_ratio * math.log(action_size)
        )
        extra = PyTreeDict(
            preferences=jnp.full(
                (self.num_objectives,), 1 / self.num_objectives, jnp.float32
            ),
            target_entropy=jnp.float32(entropy),
        )
        return AgentState(
            model_state=nnx.unpack(models),
            obs_preprocessor_state=running_statistics.init_state(obs_space.sample(key))
            if self.obs_preprocessor is not None
            else None,
            extra_state=extra,
        )

    def compute_actions(self, agent_state, sample_batch, key):
        obs = sample_batch.obs
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        raw = self.get_models(agent_state, copy=True).actor(obs)
        if self.continuous_action:
            dist = get_tanh_norm_dist(*jnp.split(raw, 2, axis=-1))
            action = dist.sample(seed=key)
        else:
            action = get_categorical_dist(raw).sample(seed=key)
        return action, PyTreeDict()

    def evaluate_actions(self, agent_state, sample_batch, key):
        obs = sample_batch.obs
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        raw = self.get_models(agent_state, copy=True).actor(obs)
        if self.continuous_action:
            dist = get_tanh_norm_dist(*jnp.split(raw, 2, axis=-1))
            action = dist.mode()
        else:
            action = get_categorical_dist(raw).mode()
        return action, PyTreeDict()

    def critic_loss(self, agent_state, sample_batch, key):
        models = self.get_models(agent_state)
        obs = sample_batch.obs
        next_obs = sample_batch.extras.env_extras.ori_obs
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )
        weights = agent_state.extra_state.preferences
        alpha = jnp.exp(models.log_alpha[...])
        raw = models.actor(next_obs)
        if self.continuous_action:
            dist = get_tanh_norm_dist(*jnp.split(raw, 2, axis=-1))
            next_action = dist.sample(seed=key)
            logp = dist.log_prob(next_action)
            next_q = scalarized_q(
                models.target_critic(next_obs, next_action), weights
            ).min(-1)
            next_v = next_q - alpha * logp
            q = scalarized_q(models.critic(obs, sample_batch.actions), weights)
        else:
            logp = jax.nn.log_softmax(raw, -1)
            probability = jnp.exp(logp)
            # (..., critics, actions, objectives) -> (..., actions).
            next_q = scalarized_q(models.target_critic(next_obs), weights).min(-2)
            next_v = (probability * (next_q - alpha * logp)).sum(-1)
            q_all = scalarized_q(models.critic(obs), weights)
            index = jnp.broadcast_to(
                sample_batch.actions[..., None, None],
                (*sample_batch.actions.shape, q_all.shape[-2], 1),
            )
            q = jnp.take_along_axis(q_all, index, -1).squeeze(-1)
        target = jax.lax.stop_gradient(
            scalarized_q(sample_batch.rewards, weights)
            + self.discount * (1 - sample_batch.extras.env_extras.termination) * next_v
        )
        return PyTreeDict(
            critic_loss=jnp.square(q - target[..., None]).sum(-1).mean(),
            q_value=q.mean(),
            target_value=target.mean(),
        )

    def actor_loss(self, agent_state, sample_batch, key):
        models = self.get_models(agent_state)
        obs = sample_batch.obs
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        alpha = jax.lax.stop_gradient(jnp.exp(models.log_alpha[...]))
        raw = models.actor(obs)
        weights = agent_state.extra_state.preferences
        if self.continuous_action:
            actor_key, entropy_key = jax.random.split(key)
            dist = get_tanh_norm_dist(*jnp.split(raw, 2, axis=-1))
            action = dist.sample(seed=actor_key)
            logp = dist.log_prob(action)
            q = scalarized_q(models.critic(obs, action), weights).min(-1)
            loss = (alpha * logp - q).mean()
            entropy = dist.entropy(seed=entropy_key).mean()
        else:
            logp = jax.nn.log_softmax(raw, -1)
            probability = jnp.exp(logp)
            q = scalarized_q(models.critic(obs), weights).min(-2)
            # Exact categorical expectation; average batches, not action count.
            loss = (probability * (alpha * logp - q)).sum(-1).mean()
            entropy = -(probability * logp).sum(-1).mean()
        return PyTreeDict(actor_loss=loss, entropy=entropy)

    def alpha_loss(self, agent_state, sample_batch, key):
        models = self.get_models(agent_state)
        obs = sample_batch.obs
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        raw = models.actor(obs)
        if self.continuous_action:
            dist = get_tanh_norm_dist(*jnp.split(raw, 2, axis=-1))
            action = dist.sample(seed=key)
            logp = dist.log_prob(action)
            residual = logp + agent_state.extra_state.target_entropy
        else:
            logp = jax.nn.log_softmax(raw, -1)
            probability = jnp.exp(logp)
            residual = (
                probability * (logp + agent_state.extra_state.target_entropy)
            ).sum(-1)
        loss = (
            -jnp.exp(models.log_alpha[...]) * jax.lax.stop_gradient(residual)
        ).mean()
        return PyTreeDict(alpha_loss=loss, alpha=jnp.exp(models.log_alpha[...]))


def _update(workflow, state):
    """Run SAC update cycles over local replay, without collecting transitions."""
    key, update_key = jax.random.split(state.key)
    update_key = rng_fold_in_axis(update_key, workflow.dp_axis_name)
    buffer = state.replay_buffer_state

    def update_critic(agent_state, optimizer_state, batch, critic_key):
        models = workflow.agent.get_models(agent_state, copy=True)

        def critic_loss(models):
            info = workflow.agent.critic_loss(
                agent_state.replace(model_state=nnx.unpack(models)), batch, critic_key
            )
            return info.critic_loss, info

        value, grads = pmean_nnx_loss_and_grad(
            critic_loss,
            workflow.dp_axis_name,
            has_aux=True,
            wrt=nnx.All(nnx.Param, nnx.PathContains("critic")),
        )(models)
        optimizer = nnx.merge(optimizer_state, copy=True)
        optimizer.update(models.critic, grads["critic"])
        return (
            value,
            agent_state.replace(model_state=nnx.unpack(models)),
            nnx.unpack(optimizer),
        )

    def update_once(carry, _):
        agent_state, opt_state, key, count = carry
        key, critic_key, actor_key, alpha_key, sample_key = jax.random.split(key, 5)
        critic_state = opt_state.critic
        if workflow.config.actor_update_interval > 1:

            def extra_critic(carry, _):
                key, agent_state, critic_state = carry
                key, sample_key, critic_key = jax.random.split(key, 3)
                batch = workflow.replay_buffer.sample(buffer, sample_key)
                _, agent_state, critic_state = update_critic(
                    agent_state, critic_state, batch, critic_key
                )
                return (key, agent_state, critic_state), None

            key, extra_key = jax.random.split(key)
            (_, agent_state, critic_state), _ = jax.lax.scan(
                extra_critic,
                (extra_key, agent_state, critic_state),
                (),
                length=workflow.config.actor_update_interval - 1,
            )
        batch = workflow.replay_buffer.sample(buffer, sample_key)
        (critic_value, critic_info), agent_state, critic_state = update_critic(
            agent_state, critic_state, batch, critic_key
        )
        models = workflow.agent.get_models(agent_state, copy=True)

        def actor_loss(models):
            info = workflow.agent.actor_loss(
                agent_state.replace(model_state=nnx.unpack(models)), batch, actor_key
            )
            return info.actor_loss, info

        (actor_value, actor_info), grads = pmean_nnx_loss_and_grad(
            actor_loss,
            workflow.dp_axis_name,
            has_aux=True,
            wrt=nnx.All(nnx.Param, nnx.PathContains("actor")),
        )(models)
        actor_optimizer = nnx.merge(opt_state.actor, copy=True)
        actor_optimizer.update(models.actor, grads["actor"])
        opt_state = opt_state.replace(
            actor=nnx.unpack(actor_optimizer), critic=critic_state
        )
        agent_state = agent_state.replace(model_state=nnx.unpack(models))
        if workflow.config.adaptive_alpha:

            def alpha_loss(models):
                info = workflow.agent.alpha_loss(
                    agent_state.replace(model_state=nnx.unpack(models)),
                    batch,
                    alpha_key,
                )
                return info.alpha_loss, info

            (alpha_value, alpha_info), grads = pmean_nnx_loss_and_grad(
                alpha_loss,
                workflow.dp_axis_name,
                has_aux=True,
                wrt=nnx.All(nnx.Param, nnx.PathContains("log_alpha")),
            )(models)
            alpha_optimizer = nnx.merge(opt_state.alpha, copy=True)
            alpha_optimizer.update(nnx.Dict(log_alpha=models.log_alpha), grads)
            opt_state = opt_state.replace(alpha=nnx.unpack(alpha_optimizer))
        else:
            alpha_value = jnp.zeros(())
            alpha_info = PyTreeDict(alpha_loss=alpha_value)
        alpha_info = alpha_info.replace(
            alpha=jnp.exp(models.log_alpha[...]), log_alpha=models.log_alpha[...]
        )
        nnx.update(
            models.target_critic,
            soft_target_update(
                nnx.state(models.target_critic, nnx.Param),
                nnx.state(models.critic, nnx.Param),
                workflow.config.tau,
            ),
        )
        return (
            agent_state.replace(model_state=nnx.unpack(models)),
            opt_state,
            key,
            count + workflow.config.actor_update_interval,
        ), (
            critic_value + actor_value + alpha_value,
            critic_info.replace(**actor_info, **alpha_info),
        )

    (agent_state, opt_state, _, updates), (loss, info) = jax.lax.scan(
        update_once,
        (
            state.agent_state,
            state.opt_state,
            update_key,
            state.metrics.training_updates,
        ),
        (),
        length=workflow.config.num_updates_per_iter,
    )
    metrics = state.metrics.replace(training_updates=updates)
    return TrainMetric(
        loss=loss.mean(), raw_loss_dict=jax.tree.map(lambda x: x.mean(0), info)
    ).all_reduce(workflow.dp_axis_name), state.replace(
        key=key, agent_state=agent_state, opt_state=opt_state, metrics=metrics
    )


class MOSACWorkflow(OffPolicyWorkflowTemplate):
    """Train one SAC policy and expose replay-only updates for population methods.

    `update` neither samples environments nor changes their states, rollout
    counters or replay contents. It changes only model/optimizer state, the root
    RNG and `metrics.training_updates`. `step` collects real transitions first.
    Each update cycle performs `actor_update_interval` critic updates, one actor
    update, an optional temperature update, then one target update, as in SAC.
    `training_updates` counts critic updates, not cycles.
    """

    @classmethod
    def name(cls):
        return "MOSAC"

    @classmethod
    def _build_from_config(cls, config):
        validate_replay_config(config)
        if (
            not 0 <= config.discount <= 1
            or not 0 <= config.tau <= 1
            or config.alpha <= 0
        ):
            raise ValueError(
                "discount/tau must lie in [0,1] and alpha must be positive"
            )
        if (
            min(
                config.actor_update_interval,
                config.num_eval_envs,
                config.eval_episodes,
                config.eval_interval,
            )
            <= 0
        ):
            raise ValueError(
                "Update frequencies and evaluation counts must be positive"
            )
        if not 0 < config.target_entropy_ratio <= 1:
            raise ValueError("target_entropy_ratio must lie in (0,1]")
        if config.optimizer.lr <= 0 or (
            config.optimizer.grad_clip_norm is not None
            and config.optimizer.grad_clip_norm < 0
        ):
            raise ValueError(
                "Learning rates must be positive and gradient clipping nonnegative"
            )
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )
        count = objective_count(env)
        if not isinstance(env.action_space, (Box, Discrete)):
            raise TypeError("MOSAC supports flat Box or Discrete actions")
        if isinstance(env.action_space, Box):
            space = env.action_space
            if len(space.shape) != 1 or not bool(
                jnp.all(
                    jnp.isfinite(space.low)
                    & jnp.isfinite(space.high)
                    & (space.high > space.low)
                )
            ):
                raise ValueError("Continuous actions require finite ordered bounds")
        if config.preferences is not None:
            weights = np.asarray(config.preferences)
            if (
                weights.shape != (count,)
                or not np.isfinite(weights).all()
                or (weights < 0).any()
                or not np.isclose(weights.sum(), 1)
            ):
                raise ValueError(
                    "preferences must be a nonnegative unit-sum objective vector"
                )
        agent = MOSACAgent(
            num_objectives=count,
            continuous_action=isinstance(env.action_space, Box),
            agent_network_cfg=config.agent_network,
            discount=config.discount,
            init_alpha=config.alpha,
            target_entropy_ratio=config.target_entropy_ratio,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
        )
        optimizer = optax.adam(config.optimizer.lr)
        if config.optimizer.grad_clip_norm:
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.optimizer.grad_clip_norm), optimizer
            )
        evaluator = Evaluator(
            env=create_env(
                config.env,
                parallel=config.num_eval_envs,
                autoreset_mode=AutoresetMode.DISABLED,
            ),
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
            discount=config.eval_discount,
        )
        replay = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            sample_batch_size=config.batch_size,
            min_sample_timesteps=max(
                config.batch_size, config.learning_start_timesteps
            ),
        )
        return cls(env, agent, optimizer, evaluator, replay, config)

    def _setup_workflow_metrics(self):
        return MultiObjectiveWorkflowMetric()

    def _setup_agent_and_optimizer(self, key):
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        if self.config.preferences is not None:
            agent_state = agent_state.replace(
                extra_state=agent_state.extra_state.replace(
                    preferences=jnp.asarray(self.config.preferences, jnp.float32)
                )
            )
        models = self.agent.get_models(agent_state, copy=True)
        opt_state = PyTreeDict(
            actor=nnx.unpack(
                nnx.Optimizer(models.actor, self.optimizer, wrt=nnx.Param)
            ),
            critic=nnx.unpack(
                nnx.Optimizer(models.critic, self.optimizer, wrt=nnx.Param)
            ),
        )
        if self.config.adaptive_alpha:
            opt_state.alpha = nnx.unpack(
                nnx.Optimizer(
                    nnx.Dict(log_alpha=models.log_alpha),
                    self.optimizer,
                    wrt=nnx.Param,
                )
            )
        return agent_state, opt_state

    def _setup_replaybuffer(self, key):
        obs = self.env.obs_space.sample(key)
        spec = SampleBatch(
            obs=obs,
            actions=self.env.action_space.sample(key),
            rewards=jnp.zeros((self.agent.num_objectives,), jnp.float32),
            extras=PyTreeDict(
                policy_extras=PyTreeDict(),
                env_extras=PyTreeDict(ori_obs=obs, termination=jnp.zeros(())),
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
        random_agent = RandomAgent()
        random_state = random_agent.init(
            self.env.obs_space, self.env.action_space, random_key
        )
        for length, random_actions, rollout_key in (
            (random_length, True, random_key),
            (policy_length, False, policy_key),
        ):
            if length == 0:
                continue
            action_fn = (
                random_agent.compute_actions
                if random_actions
                else self.agent.compute_actions
            )
            agent_state = random_state if random_actions else state.agent_state
            trajectory, env_state = rollout(
                self.env.step,
                action_fn,
                state.env_state,
                agent_state,
                rollout_key,
                length,
                env_extra_fields=("ori_obs", "termination"),
            )
            trajectory = jax.tree.map(
                jax.lax.stop_gradient,
                flatten_rollout_trajectory(clean_trajectory(trajectory)),
            )
            agent_state = state.agent_state
            if agent_state.obs_preprocessor_state is not None:
                agent_state = agent_state.replace(
                    obs_preprocessor_state=running_statistics.update(
                        agent_state.obs_preprocessor_state,
                        trajectory.obs,
                        axis_name=self.dp_axis_name,
                    )
                )
            state = state.replace(
                env_state=env_state,
                agent_state=agent_state,
                replay_buffer_state=self.replay_buffer.add(
                    state.replay_buffer_state, trajectory
                ),
                metrics=state.metrics.replace(
                    sampled_timesteps=state.metrics.sampled_timesteps
                    + psum(
                        jnp.uint32(length * self.config.num_envs), self.dp_axis_name
                    ),
                ),
            )
        if random_length + policy_length > 0:
            key, env_key = jax.random.split(key)
            env_key = rng_fold_in_axis(env_key, self.dp_axis_name)
            state = state.replace(env_state=self.env.reset(env_key))
        return state.replace(key=key)

    def update(self, state):
        """Update from an initialized local replay ring without collecting data.

        The replay ring may be broadcast read-only across a population vmap.
        Data-parallel public calls map over local replay rings. `step` invokes
        the same local computation directly inside its own shard-map boundary,
        avoiding nested maps. Only explicit gradient/statistic reductions communicate.
        """
        return _update(self, state)

    def step(self, state):
        key, rollout_key = jax.random.split(state.key)
        rollout_key = rng_fold_in_axis(rollout_key, self.dp_axis_name)
        trajectory, env_state = rollout(
            self.env.step,
            self.agent.compute_actions,
            state.env_state,
            state.agent_state,
            rollout_key,
            self.config.rollout_length,
            env_extra_fields=("ori_obs", "termination"),
        )
        episodes = trajectory.dones.sum().astype(jnp.uint32)
        trajectory = jax.tree.map(
            jax.lax.stop_gradient,
            flatten_rollout_trajectory(clean_trajectory(trajectory)),
        )
        agent_state = state.agent_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    axis_name=self.dp_axis_name,
                )
            )
        state = state.replace(
            key=key,
            env_state=env_state,
            agent_state=agent_state,
            replay_buffer_state=self.replay_buffer.add(
                state.replay_buffer_state, trajectory
            ),
            metrics=state.metrics.replace(
                iterations=state.metrics.iterations + 1,
                sampled_timesteps=state.metrics.sampled_timesteps
                + psum(
                    jnp.uint32(self.config.num_envs * self.config.rollout_length),
                    self.dp_axis_name,
                ),
                sampled_episodes=state.metrics.sampled_episodes
                + psum(episodes, self.dp_axis_name),
            ),
        )
        return _update(self, state)

    def evaluate(self, state):
        key, eval_key = jax.random.split(state.key)
        eval_key = rng_fold_in_axis(eval_key, self.dp_axis_name)
        evaluation = self.evaluator.evaluate(
            state.agent_state, eval_key, self.config.eval_episodes
        )
        weights = state.agent_state.extra_state.preferences
        return MultiObjectiveEvaluateMetric(
            preferences=weights,
            episode_returns=evaluation.episode_returns.mean(0),
            scalarized_returns=(evaluation.episode_returns.mean(0) * weights).sum(),
            episode_lengths=evaluation.episode_lengths.mean(),
        ).all_reduce(self.dp_axis_name), state.replace(key=key)

    @classmethod
    def enable_shmap(cls):
        super().enable_shmap()
        specs = State(
            key=jax.P(),
            metrics=jax.P(),
            agent_state=jax.P(),
            opt_state=jax.P(),
            env_state=jax.P(DP_AXIS_NAME),
            replay_buffer_state=jax.P(DP_AXIS_NAME),
        )
        update_fn = cls.update

        def sharded_update(self, state):
            with _shard_map_replay_buffer_context(state) as local:
                metrics, next_state = update_fn(self, local)
                local.update(next_state)
            return metrics, local

        cls.update = jax.shard_map(
            sharded_update,
            in_specs=(None, specs),
            out_specs=(jax.P(), specs),
        )

    @classmethod
    def enable_jit(cls):
        super().enable_jit()
        cls.update = jax.jit(cls.update, static_argnums=(0,))

    def learn(self, state):
        quantum = self.config.num_envs * self.config.rollout_length * self.num_devices
        while int(state.metrics.sampled_timesteps) < self.config.total_timesteps:
            remaining = self.config.total_timesteps - int(
                state.metrics.sampled_timesteps
            )
            count = min(self.config.fold_iters, math.ceil(remaining / quantum))
            if count == self.config.fold_iters:
                train, state = self._multi_steps(state)
            else:
                for _ in range(count):
                    train, state = self.step(state)
            iteration = int(state.metrics.iterations)
            self.recorder.write(state.metrics.to_local_dict(), iteration)
            self.recorder.write(train.to_local_dict(), iteration)
            final = int(state.metrics.sampled_timesteps) >= self.config.total_timesteps
            if iteration % self.config.eval_interval == 0 or final:
                evaluation, state = self.evaluate(state)
                self.recorder.write(
                    add_prefix(evaluation.to_local_dict(), "eval"), iteration
                )
            saved = (
                state
                if self.config.save_replay_buffer
                else state.replace(replay_buffer_state=None)
            )
            self.checkpoint_manager.save(iteration, saved, force=final)
        return state
