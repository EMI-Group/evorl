"""Fixed-preference multi-objective PPO, also used by population methods.

Vector rewards and values remain separate until the policy advantage is
scalarized. See MORL-Baselines' `single_policy/ser/mo_ppo.py`.
"""

import math
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import DictConfig

from evorl.agent import Agent, AgentState
from evorl.distributed import pmean, pmean_nnx_loss_and_grad, psum
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.envs import AutoresetMode, Box, Discrete, create_env
from evorl.evaluators import Evaluator
from evorl.metrics import TrainMetric
from evorl.networks import make_mlp, make_policy_network
from evorl.recorders import add_prefix
from evorl.rollout import rollout
from evorl.types import PyTreeDict, pytree_field
from evorl.utils import running_statistics
from evorl.utils.jax_utils import rng_fold_in_axis
from evorl.utils.rl_toolkits import (
    approximate_kl,
    compute_gae,
    flatten_rollout_trajectory,
)
from evorl.workflows import OnPolicyWorkflow

from .common import (
    MultiObjectiveEvaluateMetric,
    MultiObjectiveWorkflowMetric,
    objective_count,
)


class MOPPOAgent(Agent):
    """A fixed-preference policy and vector critic with functional NNX state.

    Preferences live in `AgentState.extra_state.preferences`; replacing that
    vector changes the policy objective without rebuilding the static agent.
    The policy itself is not preference-conditioned.
    """

    num_objectives: int = pytree_field(static=True)
    continuous_action: bool = pytree_field(static=True)
    agent_network_cfg: DictConfig = pytree_field(static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)
    clip_epsilon: float = 0.2
    normalize_gae: bool = True
    clip_value_loss: bool = True

    def init(self, obs_space, action_space, key):
        actor_key, critic_key = jax.random.split(key)
        cfg = self.agent_network_cfg
        actor_size = (
            action_space.shape[0] * 2 if self.continuous_action else action_space.n
        )
        models = nnx.Dict(
            policy=make_policy_network(
                obs_space.shape[0],
                actor_size,
                hidden_layer_sizes=cfg.actor_hidden_layer_sizes,
                activation=jax.nn.tanh,
                rngs=nnx.Rngs(actor_key),
            ),
            value=make_mlp(
                obs_space.shape[0],
                (*cfg.critic_hidden_layer_sizes, self.num_objectives),
                activation=jax.nn.tanh,
                rngs=nnx.Rngs(critic_key),
            ),
        )
        extra = PyTreeDict(
            preferences=jnp.ones(self.num_objectives) / self.num_objectives
        )
        return AgentState(
            model_state=nnx.unpack(models),
            obs_preprocessor_state=running_statistics.init_state(
                jnp.zeros(obs_space.shape, jnp.float32)
            )
            if self.obs_preprocessor is not None
            else None,
            extra_state=extra,
        )

    def compute_actions(self, agent_state, sample_batch, key):
        obs = sample_batch.obs
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        raw = self.get_models(agent_state, copy=True).policy(obs)
        if self.continuous_action:
            dist = get_tanh_norm_dist(*jnp.split(raw, 2, -1))
            action = dist.sample(seed=key)
            logp = dist.log_prob(action)
        else:
            dist = get_categorical_dist(raw)
            action = dist.sample(seed=key)
            logp = dist.log_prob(action)
        return action, PyTreeDict(logp=logp)

    def evaluate_actions(self, agent_state, sample_batch, key):
        obs = sample_batch.obs
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        raw = self.get_models(agent_state, copy=True).policy(obs)
        if self.continuous_action:
            dist = get_tanh_norm_dist(*jnp.split(raw, 2, -1))
            action = dist.mode()
        else:
            action = get_categorical_dist(raw).mode()
        return action, PyTreeDict()

    def compute_values(self, agent_state, sample_batch):
        obs = sample_batch.obs
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        return self.get_models(agent_state, copy=True).value(obs)

    def loss(self, agent_state, sample_batch, key, *, axis_name=None):
        """Compute PPO losses, normalizing advantages over the global minibatch.

        Args:
            agent_state: Functional model and observation-preprocessor state.
            sample_batch: Local minibatch, including the autoreset mask.
            key: Random key for entropy estimation.
            axis_name: Data-parallel Manual axis for valid-sample counts and
                advantage statistics, or `None` for a local minibatch.
                Population axes must not be used.

        Returns:
            Loss terms weighted for the subsequent device mean, excluding
            reset-only transitions from statistics and losses.
        """
        models = self.get_models(agent_state)
        obs = sample_batch.obs
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        mask = jnp.logical_not(sample_batch.extras.env_extras.autoreset)
        mean_count = pmean(mask.sum(dtype=jnp.float32), axis_name)
        denominator = jnp.where(mean_count > 0, mean_count, 1.0)
        raw = models.policy(obs)
        if self.continuous_action:
            dist = get_tanh_norm_dist(*jnp.split(raw, 2, -1))
            logp = dist.log_prob(sample_batch.actions)
            entropy = dist.entropy(seed=key)
        else:
            dist = get_categorical_dist(raw)
            logp = dist.log_prob(sample_batch.actions)
            entropy = dist.entropy()
        advantage = (
            sample_batch.extras.advantages * agent_state.extra_state.preferences
        ).sum(-1)
        if self.normalize_gae:
            mean = pmean(advantage.sum(where=mask), axis_name) / denominator
            variance = (
                pmean(jnp.square(advantage - mean).sum(where=mask), axis_name)
                / denominator
            )
            advantage = (advantage - mean) / (jnp.sqrt(variance) + 1e-8)
        logratio = logp - sample_batch.extras.policy_extras.logp
        ratio = jnp.exp(logratio)
        clipped = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        actor_loss = (
            -jnp.minimum(ratio * advantage, clipped * advantage).sum(where=mask)
            / denominator
        )
        values = models.value(obs)
        squared = (values - sample_batch.extras.v_targets) ** 2
        if self.clip_value_loss:
            old = sample_batch.extras.values
            clipped_values = old + jnp.clip(
                values - old, -self.clip_epsilon, self.clip_epsilon
            )
            squared = jnp.maximum(
                squared, (clipped_values - sample_batch.extras.v_targets) ** 2
            )
        return PyTreeDict(
            actor_loss=actor_loss,
            critic_loss=0.5 * squared.mean(-1).sum(where=mask) / denominator,
            actor_entropy=entropy.sum(where=mask) / denominator,
            approx_kl=approximate_kl(logratio, axis=()).sum(where=mask) / denominator,
        )


class MOPPOWorkflow(OnPolicyWorkflow):
    """Train one policy for a fixed linear scalarization of vector returns."""

    @classmethod
    def name(cls):
        return "MOPPO"

    @classmethod
    def _rescale_config(cls, config):
        super()._rescale_config(config)
        devices = jax.local_device_count()
        for name in ("num_envs", "minibatch_size"):
            if config[name] < devices or config[name] % devices:
                raise ValueError(f"{name} must be divisible by the device count")
            config[name] //= devices

    @classmethod
    def _build_from_config(cls, config):
        if not 0 <= config.discount <= 1 or not 0 <= config.gae_lambda <= 1:
            raise ValueError("discount and gae_lambda must lie in [0, 1]")
        if not 0 <= config.get("eval_discount", 1.0) <= 1:
            raise ValueError("eval_discount must lie in [0, 1]")
        if config.clip_epsilon < 0 or config.optimizer.lr <= 0:
            raise ValueError(
                "clip_epsilon must be nonnegative and learning rate positive"
            )
        if (
            config.optimizer.grad_clip_norm is not None
            and config.optimizer.grad_clip_norm < 0
        ):
            raise ValueError("grad_clip_norm must be nonnegative")
        if min(config.num_eval_envs, config.eval_episodes, config.eval_interval) <= 0:
            raise ValueError("Evaluation counts and interval must be positive")
        if (
            min(
                config.num_envs,
                config.rollout_length,
                config.minibatch_size,
                config.reuse_rollout_epochs,
            )
            <= 0
        ):
            raise ValueError("Rollout, minibatch and epoch sizes must be positive")
        if (config.num_envs * config.rollout_length) % config.minibatch_size:
            raise ValueError("minibatch_size must divide num_envs * rollout_length")
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.ENVPOOL,
        )
        count = objective_count(env)
        if not isinstance(env.action_space, (Box, Discrete)):
            raise TypeError("MOPPO requires a continuous Box or Discrete action space")
        if isinstance(env.action_space, Box):
            space = env.action_space
            if len(space.shape) != 1 or not bool(
                jnp.all(
                    jnp.isfinite(space.low)
                    & jnp.isfinite(space.high)
                    & (space.high > space.low)
                )
            ):
                raise ValueError(
                    "MOPPO requires finite, ordered one-dimensional action bounds"
                )
        preferences = config.preferences
        if preferences is not None:
            weights = np.asarray(preferences)
            if (
                weights.shape != (count,)
                or not np.isfinite(weights).all()
                or (weights < 0).any()
                or not np.isclose(weights.sum(), 1)
            ):
                raise ValueError(
                    "preferences must be a nonnegative unit-sum objective vector"
                )
        agent = MOPPOAgent(
            num_objectives=count,
            continuous_action=isinstance(env.action_space, Box),
            agent_network_cfg=config.agent_network,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
            clip_epsilon=config.clip_epsilon,
            normalize_gae=config.normalize_gae,
            clip_value_loss=config.clip_value_loss,
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
            discount=config.get("eval_discount", 1.0),
        )
        return cls(env, agent, optimizer, evaluator, config)

    def _setup_agent_and_optimizer(self, key):
        state, optimizer = super()._setup_agent_and_optimizer(key)
        if self.config.preferences is not None:
            state = state.replace(
                extra_state=state.extra_state.replace(
                    preferences=jnp.asarray(self.config.preferences, jnp.float32)
                )
            )
        return state, optimizer

    def _setup_workflow_metrics(self):
        return MultiObjectiveWorkflowMetric()

    def step(self, state):
        key, rollout_key, learn_key = jax.random.split(state.key, 3)
        rollout_key, learn_key = rng_fold_in_axis(
            (rollout_key, learn_key), self.dp_axis_name
        )
        trajectory, env_state = rollout(
            self.env.step,
            self.agent.compute_actions,
            state.env_state,
            state.agent_state,
            rollout_key,
            self.config.rollout_length,
            env_extra_fields=("autoreset", "termination"),
        )
        # ENVPOOL preserves the final observation until the next reset-only step,
        # so adjacent values provide the true bootstrap observation at truncation.
        observations = jax.tree.map(
            lambda obs, next_obs: jnp.concatenate([obs, next_obs[-1:]], axis=0),
            trajectory.obs,
            trajectory.next_obs,
        )
        # Bootstrap uses exactly the observation statistics that generated actions.
        values = self.agent.compute_values(
            state.agent_state, trajectory.replace(obs=observations)
        )
        targets, advantages = compute_gae(
            rewards=trajectory.rewards,
            values=values,
            dones=trajectory.dones[..., None],
            terminations=trajectory.extras.env_extras.termination[..., None],
            discount=self.config.discount,
            gae_lambda=self.config.gae_lambda,
        )
        trajectory.extras.values = values[:-1]
        trajectory.extras.v_targets = targets
        trajectory.extras.advantages = advantages
        trajectory = jax.tree.map(
            jax.lax.stop_gradient, flatten_rollout_trajectory(trajectory)
        )
        num_minibatches = (
            self.config.rollout_length
            * self.config.num_envs
            // self.config.minibatch_size
        )

        def loss_fn(models, agent_state, batch, key):
            losses = self.agent.loss(
                agent_state.replace(model_state=nnx.unpack(models)),
                batch,
                key,
                axis_name=self.dp_axis_name,
            )
            loss = sum(
                self.config.loss_weights[name] * losses[name]
                for name in self.config.loss_weights
            )
            return loss, losses

        loss_grad = pmean_nnx_loss_and_grad(loss_fn, self.dp_axis_name, has_aux=True)

        def minibatch_step(carry, batch):
            agent_state, opt_state, key = carry
            key, loss_key = jax.random.split(key)
            models = nnx.merge(agent_state.model_state, copy=True)
            optimizer = nnx.merge(opt_state, copy=True)
            (loss, losses), grads = loss_grad(models, agent_state, batch, loss_key)
            optimizer.update(models, grads)
            return (
                agent_state.replace(model_state=nnx.unpack(models)),
                nnx.unpack(optimizer),
                key,
            ), (loss, losses)

        def epoch_step(carry, _):
            agent_state, opt_state, key = carry
            key, shuffle_key = jax.random.split(key)
            indices = jax.random.permutation(
                shuffle_key, self.config.num_envs * self.config.rollout_length
            )
            minibatches = jax.tree.map(
                lambda x: x[indices].reshape(
                    num_minibatches, self.config.minibatch_size, *x.shape[1:]
                ),
                trajectory,
            )
            carry, losses = jax.lax.scan(
                minibatch_step, (agent_state, opt_state, key), minibatches
            )
            return carry, jax.tree.map(lambda x: x.mean(0), losses)

        (agent_state, opt_state, _), (losses, raw_losses) = jax.lax.scan(
            epoch_step,
            (state.agent_state, state.opt_state, learn_key),
            None,
            length=self.config.reuse_rollout_epochs,
        )
        # Keep normalization fixed throughout PPO's reuse of the behavior batch.
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    axis_name=self.dp_axis_name,
                )
            )
        metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps
            + psum(
                jnp.uint32(self.config.num_envs * self.config.rollout_length),
                self.dp_axis_name,
            ),
            sampled_episodes=state.metrics.sampled_episodes
            + psum(trajectory.dones.sum().astype(jnp.uint32), self.dp_axis_name),
            training_updates=state.metrics.training_updates
            + self.config.reuse_rollout_epochs * num_minibatches,
            iterations=state.metrics.iterations + 1,
        )
        train = TrainMetric(
            loss=losses.mean(),
            raw_loss_dict=jax.tree.map(lambda x: x.mean(0), raw_losses),
        ).all_reduce(self.dp_axis_name)
        return train, state.replace(
            key=key,
            agent_state=agent_state,
            opt_state=opt_state,
            env_state=env_state,
            metrics=metrics,
        )

    def evaluate(self, state):
        key, eval_key = jax.random.split(state.key)
        eval_key = rng_fold_in_axis(eval_key, self.dp_axis_name)
        metrics = self.evaluator.evaluate(
            state.agent_state, eval_key, self.config.eval_episodes
        )
        returns = pmean(metrics.episode_returns.mean(0), self.dp_axis_name)
        preferences = state.agent_state.extra_state.preferences
        result = MultiObjectiveEvaluateMetric(
            preferences=preferences,
            episode_returns=returns,
            scalarized_returns=jnp.sum(returns * preferences),
            episode_lengths=pmean(metrics.episode_lengths.mean(), self.dp_axis_name),
        )
        return result, state.replace(key=key)

    def learn(self, state):
        steps = self.config.rollout_length * self.config.num_envs * self.num_devices
        final = math.ceil(self.config.total_timesteps / steps)
        for _ in range(int(state.metrics.iterations), final):
            train, state = self.step(state)
            iteration = int(state.metrics.iterations)
            self.recorder.write(state.metrics.to_local_dict(), iteration)
            self.recorder.write(train.to_local_dict(), iteration)
            if iteration % self.config.eval_interval == 0 or iteration == final:
                evaluation, state = self.evaluate(state)
                self.recorder.write(
                    add_prefix(evaluation.to_local_dict(), "eval"), iteration
                )
            self.checkpoint_manager.save(iteration, state, force=iteration == final)
        return state
