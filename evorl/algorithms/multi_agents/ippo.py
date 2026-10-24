"""Independent PPO with a separate local actor and value function per agent."""

import math

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from evorl.agent import AgentState
from evorl.distributed import psum, pmean_nnx_loss_and_grad
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators import MultiAgentEvaluator
from evorl.metrics import EvaluateMetric, TrainMetric
from evorl.networks import make_policy_network, make_v_network
from evorl.recorders import add_prefix
from evorl.rollout import multi_agent_rollout
from evorl.types import PyTreeDict, pytree_field
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.rl_toolkits import approximate_kl, flatten_rollout_trajectory
from evorl.workflows import OnPolicyWorkflow

from .common import (
    MultiAgentAgent,
    agent_layout,
    centralized_features,
    init_preprocessing,
    local_features,
    masked_logits,
    masked_mean,
    normalize_action,
    scale_action,
    update_preprocessing,
)


def compute_agent_gae(
    rewards, values, next_values, dones, terminations, active, discount, gae_lambda
):
    """Compute per-agent GAE, excluding reset-only and inactive-agent steps.

    Natural termination suppresses bootstrap. Time truncation retains one-step
    bootstrap but stops the trace. Inactive agents contribute no advantage.
    Under ENVPOOL autoreset, adjacent observations retain terminal values.
    """

    def step(carry, transition):
        reward, value, next_value, done, terminated, valid = transition
        delta = reward + discount * (1 - terminated) * next_value - value
        advantage = jnp.where(
            valid, delta + discount * gae_lambda * (1 - done) * carry, 0
        )
        return advantage, advantage

    _, advantages = jax.lax.scan(
        step,
        jnp.zeros_like(values[0]),
        (rewards, values, next_values, dones, terminations, active),
        reverse=True,
    )
    return advantages + values, advantages


class IPPOAgent(MultiAgentAgent):
    """Agent-keyed feedforward PPO policies, with optional centralized values.

    Networks and optimizers are independent per agent. IPPO values use local
    observations; MAPPO values use global information plus the local observation.
    Rewards are kept per agent, including for mixed cooperative/competitive tasks.
    """

    centralized_critic: bool = pytree_field(default=False, static=True)
    clip_epsilon: float = 0.2
    value_clip_epsilon: float | None = 0.2

    def init(self, obs_space, action_space, key):
        models = nnx.Dict()
        keys = jax.random.split(key, 2 * len(self.agents))
        local_sizes = [
            sum(map(math.prod, shapes)) for shapes in self.observation_shapes
        ]
        global_size = (
            sum(map(math.prod, self.global_observation_shapes))
            if self.global_observation_shapes is not None
            else sum(local_sizes)
        )
        cfg = self.agent_network_cfg
        for index, name in enumerate(self.agents):
            models[name] = nnx.Dict(
                actor=make_policy_network(
                    input_size=local_sizes[index],
                    action_size=self.action_sizes[index]
                    * (2 if self.continuous_actions[index] else 1),
                    hidden_layer_sizes=cfg.actor_hidden_layer_sizes,
                    rngs=nnx.Rngs(keys[2 * index]),
                ),
                value=make_v_network(
                    input_size=local_sizes[index]
                    + (global_size if self.centralized_critic else 0),
                    hidden_layer_sizes=cfg.critic_hidden_layer_sizes,
                    rngs=nnx.Rngs(keys[2 * index + 1]),
                ),
            )
        return AgentState(
            model_state=nnx.unpack(models),
            obs_preprocessor_state=init_preprocessing(self),
        )

    def compute_actions(self, agent_state, sample_batch, key):
        models = self.get_models(agent_state, copy=True)
        features = local_features(self, agent_state, sample_batch.obs)
        actions, logp = PyTreeDict(), PyTreeDict()
        keys = jax.random.split(key, len(self.agents))
        for index, name in enumerate(self.agents):
            outputs = models[name].actor(features[name])
            if self.continuous_actions[index]:
                dist = get_tanh_norm_dist(*jnp.split(outputs, 2, axis=-1))
                normalized = dist.sample(seed=keys[index])
                actions[name] = scale_action(
                    normalized, self.action_bounds[index], self.action_shapes[index]
                )
                low, high = map(jnp.asarray, self.action_bounds[index])
                logp[name] = dist.log_prob(normalized) - jnp.log((high - low) / 2).sum()
            else:
                dist = get_categorical_dist(
                    masked_logits(outputs, sample_batch.obs[name])
                )
                actions[name] = dist.sample(seed=keys[index])
                logp[name] = dist.log_prob(actions[name])
        return actions, PyTreeDict(logp=logp)

    def evaluate_actions(self, agent_state, sample_batch, key):
        models = self.get_models(agent_state, copy=True)
        features = local_features(self, agent_state, sample_batch.obs)
        actions = PyTreeDict()
        for index, name in enumerate(self.agents):
            outputs = models[name].actor(features[name])
            if self.continuous_actions[index]:
                dist = get_tanh_norm_dist(*jnp.split(outputs, 2, axis=-1))
                actions[name] = scale_action(
                    dist.mode(), self.action_bounds[index], self.action_shapes[index]
                )
            else:
                actions[name] = jnp.argmax(
                    masked_logits(outputs, sample_batch.obs[name]), axis=-1
                )
        return actions, PyTreeDict()

    def compute_values(self, agent_state, sample_batch):
        models = self.get_models(agent_state, copy=True)
        features = local_features(self, agent_state, sample_batch.obs)
        global_features = None
        if self.centralized_critic:
            global_features = centralized_features(
                self, agent_state, sample_batch.obs, sample_batch.extras.global_obs
            )
        return PyTreeDict(
            {
                name: models[name].value(
                    jnp.concatenate([global_features, features[name]], axis=-1)
                    if self.centralized_critic
                    else features[name]
                )
                for name in self.agents
            }
        )

    def loss(self, agent_state, sample_batch, key, agent_id, axis_name=None):
        models = self.get_models(agent_state)
        index = self.agents.index(agent_id)
        features = local_features(self, agent_state, sample_batch.obs)
        inputs = features[agent_id]
        if self.centralized_critic:
            global_features = centralized_features(
                self, agent_state, sample_batch.obs, sample_batch.extras.global_obs
            )
            inputs = jnp.concatenate([global_features, inputs], axis=-1)
        values = models[agent_id].value(inputs)
        targets = sample_batch.extras.value_targets[agent_id]
        value_error = optax.squared_error(values, targets)
        if self.value_clip_epsilon is not None:
            old_values = sample_batch.extras.old_values[agent_id]
            clipped_values = old_values + jnp.clip(
                values - old_values, -self.value_clip_epsilon, self.value_clip_epsilon
            )
            value_error = jnp.maximum(
                value_error, optax.squared_error(clipped_values, targets)
            )
        outputs = models[agent_id].actor(features[agent_id])
        if self.continuous_actions[index]:
            dist = get_tanh_norm_dist(*jnp.split(outputs, 2, axis=-1))
            actions = normalize_action(
                sample_batch.actions[agent_id],
                self.action_bounds[index],
                self.action_shapes[index],
            )
            low, high = map(jnp.asarray, self.action_bounds[index])
            logp = dist.log_prob(actions) - jnp.log((high - low) / 2).sum()
            entropy = dist.entropy(seed=key) + jnp.log((high - low) / 2).sum()
        else:
            dist = get_categorical_dist(
                masked_logits(outputs, sample_batch.obs[agent_id])
            )
            logp = dist.log_prob(sample_batch.actions[agent_id])
            entropy = dist.entropy()
        ratio = jnp.exp(logp - sample_batch.extras.policy_extras.logp[agent_id])
        advantages = sample_batch.extras.advantages[agent_id]
        objective = jnp.minimum(
            ratio * advantages,
            jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages,
        )
        mask = sample_batch.extras.active[agent_id]
        return PyTreeDict(
            actor_loss=-masked_mean(objective, mask, axis_name),
            critic_loss=masked_mean(value_error, mask, axis_name),
            actor_entropy=masked_mean(entropy, mask, axis_name),
            approx_kl=masked_mean(
                approximate_kl(
                    logp - sample_batch.extras.policy_extras.logp[agent_id], axis=()
                ),
                mask,
                axis_name,
            ),
        )


class IPPOWorkflow(OnPolicyWorkflow):
    """Independent PPO, with one optimizer and gradient clipping per agent."""

    centralized_critic = False

    @classmethod
    def name(cls):
        return "IPPO"

    @classmethod
    def _rescale_config(cls, config):
        super()._rescale_config(config)
        for name in ("num_envs", "minibatch_size"):
            if config[name] % jax.local_device_count():
                raise ValueError(f"{name} must be divisible by the device count")
            config[name] //= jax.local_device_count()

    @classmethod
    def _build_from_config(cls, config):
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.ENVPOOL,
        )
        layout = agent_layout(env)
        batch_size = config.rollout_length * config.num_envs
        if (
            batch_size <= 0
            or config.minibatch_size <= 0
            or batch_size % config.minibatch_size
        ):
            raise ValueError(
                "rollout_length * num_envs must be a positive multiple of minibatch_size"
            )
        if config.reuse_rollout_epochs <= 0 or config.total_timesteps < 0:
            raise ValueError(
                "reuse_rollout_epochs must be positive and total_timesteps nonnegative"
            )
        agent = IPPOAgent(
            **layout,
            agent_network_cfg=config.agent_network,
            centralized_critic=cls.centralized_critic,
            clip_epsilon=config.clip_epsilon,
            value_clip_epsilon=config.value_clip_epsilon,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
        )
        optimizer = optax.adam(config.optimizer.lr)
        if config.optimizer.grad_clip_norm:
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.optimizer.grad_clip_norm), optimizer
            )
        eval_env = create_env(
            config.env,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )
        evaluator = MultiAgentEvaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )
        return cls(env, agent, optimizer, evaluator, config)

    def _setup_agent_and_optimizer(self, key):
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        models = nnx.merge(agent_state.model_state, copy=True)
        opt_state = PyTreeDict(
            {
                name: nnx.unpack(
                    nnx.Optimizer(models[name], self.optimizer, wrt=nnx.Param)
                )
                for name in self.agent.agents
            }
        )
        return agent_state, opt_state

    def evaluate(self, state):
        key, eval_key = jax.random.split(state.key)
        if self.is_multi_device:
            eval_key = jax.random.fold_in(
                eval_key, jax.lax.axis_index(self.dp_axis_name)
            )
        metrics = self.evaluator.evaluate(
            state.agent_state, eval_key, self.config.eval_episodes
        )
        return EvaluateMetric(
            episode_returns=jax.tree.map(jnp.mean, metrics.episode_returns),
            episode_lengths=metrics.episode_lengths.mean(),
        ).all_reduce(self.dp_axis_name), state.replace(key=key)

    def step(self, state):
        key, rollout_key, learn_key = jax.random.split(state.key, 3)
        if self.is_multi_device:
            rank = jax.lax.axis_index(self.dp_axis_name)
            rollout_key, learn_key = (
                jax.random.fold_in(k, rank) for k in (rollout_key, learn_key)
            )
        trajectory, env_state = multi_agent_rollout(
            self.env,
            self.agent.compute_actions,
            state.env_state,
            state.agent_state,
            rollout_key,
            self.config.rollout_length,
        )
        # ENVPOOL leaves terminal observations in the trajectory. Evaluate the
        # T + 1 adjacent observations once, including centralized observations.
        observations = jax.tree.map(
            lambda obs, final: jnp.concatenate([obs, final[None]], axis=0),
            trajectory.obs,
            env_state.obs,
        )
        global_obs = jax.tree.map(
            lambda obs, final: jnp.concatenate([obs, final[None]], axis=0),
            trajectory.extras.global_obs,
            env_state.info.get("global_obs"),
        )
        values = self.agent.compute_values(
            state.agent_state,
            trajectory.replace(
                obs=observations,
                extras=trajectory.extras.replace(global_obs=global_obs),
            ),
        )
        old_values = jax.tree.map(lambda x: x[:-1], values)
        next_values = jax.tree.map(lambda x: x[1:], values)
        targets, advantages = PyTreeDict(), PyTreeDict()
        for name in self.agent.agents:
            targets[name], advantage = compute_agent_gae(
                trajectory.rewards[name],
                old_values[name],
                next_values[name],
                trajectory.dones[name],
                trajectory.extras.terminations[name],
                trajectory.extras.active[name],
                self.config.discount,
                self.config.gae_lambda,
            )
            if self.config.normalize_gae:
                mask = trajectory.extras.active[name]
                count = psum(mask.sum(), self.dp_axis_name)
                mean = psum(
                    jnp.where(mask, advantage, 0).sum(), self.dp_axis_name
                ) / jnp.maximum(count, 1)
                variance = psum(
                    jnp.where(mask, (advantage - mean) ** 2, 0).sum(), self.dp_axis_name
                ) / jnp.maximum(count, 1)
                advantage = (advantage - mean) / jnp.sqrt(variance + 1e-8)
            advantages[name] = advantage
        trajectory = trajectory.replace(
            extras=trajectory.extras.replace(
                old_values=old_values, value_targets=targets, advantages=advantages
            )
        )
        episode_done = trajectory.extras.episode_done
        trajectory = tree_stop_gradient(flatten_rollout_trajectory(trajectory))
        agent_state, opt_state = state.agent_state, state.opt_state

        def minibatch(carry, batch):
            agent_state, opt_state, key = carry
            models = nnx.merge(agent_state.model_state, copy=True)
            losses, total = PyTreeDict(), jnp.float32(0)
            for name in self.agent.agents:
                key, loss_key = jax.random.split(key)

                def loss_fn(models, loss_key, name=name):
                    losses = self.agent.loss(
                        agent_state.replace(model_state=nnx.unpack(models)),
                        batch,
                        loss_key,
                        name,
                        self.dp_axis_name,
                    )
                    loss = sum(
                        self.config.loss_weights[field] * losses[field]
                        for field in self.config.loss_weights
                    )
                    return loss, losses

                (loss, info), grads = pmean_nnx_loss_and_grad(
                    loss_fn,
                    self.dp_axis_name,
                    has_aux=True,
                    wrt=lambda path, value, name=name: (
                        path[0] == name and isinstance(value, nnx.Param)
                    ),
                )(models, loss_key)
                optimizer = nnx.merge(opt_state[name], copy=True)
                optimizer.update(models[name], grads[name])
                opt_state = opt_state.replace(**{name: nnx.unpack(optimizer)})
                losses[name] = info
                total += loss
            return (
                agent_state.replace(model_state=nnx.unpack(models)),
                opt_state,
                key,
            ), (total, losses)

        def epoch(carry, _):
            agent_state, opt_state, key = carry
            key, shuffle_key = jax.random.split(key)
            indices = jax.random.permutation(
                shuffle_key, self.config.rollout_length * self.config.num_envs
            )
            batches = jax.tree.map(
                lambda x: x[indices].reshape(
                    (-1, self.config.minibatch_size, *x.shape[1:])
                ),
                trajectory,
            )
            carry, metrics = jax.lax.scan(
                minibatch, (agent_state, opt_state, key), batches
            )
            return carry, jax.tree.map(lambda x: x.mean(axis=0), metrics)

        (agent_state, opt_state, _), (loss, losses) = jax.lax.scan(
            epoch,
            (agent_state, opt_state, learn_key),
            (),
            length=self.config.reuse_rollout_epochs,
        )
        agent_state = update_preprocessing(
            self.agent, agent_state, trajectory, self.dp_axis_name
        )
        metrics = state.metrics.replace(
            iterations=state.metrics.iterations + 1,
            sampled_timesteps=state.metrics.sampled_timesteps
            + psum(
                jnp.uint32(self.config.rollout_length * self.config.num_envs),
                self.dp_axis_name,
            ),
            sampled_episodes=state.metrics.sampled_episodes
            + psum(episode_done.sum().astype(jnp.uint32), self.dp_axis_name),
        )
        train_metrics = TrainMetric(
            loss=loss.mean(),
            raw_loss_dict=jax.tree.map(lambda x: x.mean(axis=0), losses),
        ).all_reduce(self.dp_axis_name)
        return train_metrics, state.replace(
            key=key,
            agent_state=agent_state,
            opt_state=opt_state,
            env_state=env_state,
            metrics=metrics,
        )

    def learn(self, state):
        samples_per_step = (
            self.config.rollout_length * self.config.num_envs * self.num_devices
        )
        remaining = max(
            0,
            math.ceil(
                (self.config.total_timesteps - int(state.metrics.sampled_timesteps))
                / samples_per_step
            ),
        )
        final_iteration = int(state.metrics.iterations) + remaining
        for _ in range(remaining):
            train_metrics, state = self.step(state)
            iteration = int(state.metrics.iterations)
            self.recorder.write(state.metrics.to_local_dict(), iteration)
            self.recorder.write(train_metrics.to_local_dict(), iteration)
            if (
                iteration % self.config.eval_interval == 0
                or iteration == final_iteration
            ):
                metrics, state = self.evaluate(state)
                self.recorder.write(
                    add_prefix(metrics.to_local_dict(), "eval"), iteration
                )
            self.checkpoint_manager.save(
                iteration, state, force=iteration == final_iteration
            )
        return state
