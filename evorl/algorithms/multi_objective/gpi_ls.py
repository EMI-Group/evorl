"""GPI Linear Support with a discrete-action successor-feature network.

The DDQN variant follows MORL-Baselines' GPI-LS implementation and Alegre
et al. (AAMAS 2023), https://arxiv.org/abs/2301.07784. Training and evaluation
are JAX transformations; corner enumeration and task selection run on the host
at learning-stage boundaries. No model-based GPI-PD component is used.
"""

from itertools import combinations
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import DictConfig

from evorl.agent import Agent, AgentState, RandomAgent
from evorl.algorithms.offpolicy_utils import clean_trajectory
from evorl.distributed import pmean, pmean_nnx_loss_and_grad, psum
from evorl.distributed.sharding import mark_varying
from evorl.envs import AutoresetMode, Discrete, create_env
from evorl.evaluators import Evaluator
from evorl.metrics import TrainMetric
from evorl.networks import make_multi_objective_discrete_q_network
from evorl.replay_buffers import ReplayBuffer
from evorl.sample_batch import SampleBatch
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


def gpi_actions(values, preferences, support_mask):
    """Maximize scalarized successor features over policies and actions.

    Args:
        values: Successor features with shape `(..., policies, actions, objectives)`.
        preferences: Query weights with shape `(..., objectives)`.
        support_mask: Boolean mask selecting the valid policies.

    Returns:
        Greedy actions with shape `(...)`. The same query preference evaluates
        every candidate policy; policy-conditioning weights are not utilities.
    """
    scores = (values * preferences[..., None, None, :]).sum(-1)
    scores = jnp.where(support_mask[..., None], scores, -jnp.inf)
    return scores.max(-2).argmax(-1)


def linear_support_corners(values, num_objectives):
    """Enumerate vertices of the convex-coverage utility surface on the simplex.

    The host calculation intersects the simplex equality with active value
    planes and simplex boundaries. It avoids optional polyhedral packages and
    supports more than two objectives. Runtime grows combinatorially with the
    number of objectives and coverage vectors, as exact vertex enumeration does.

    Args:
        values: Coverage return vectors with shape `(policies, objectives)`.
        num_objectives: Number of reward coordinates.

    Returns:
        Unique corner preferences, including the simplex extrema.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1, num_objectives)
    if not np.isfinite(values).all():
        raise ValueError("Coverage values must be finite.")
    if len(values) == 0:
        return np.eye(num_objectives, dtype=np.float32)
    # Variables are (weight_1, ..., weight_m, utility). Utility >= value @ w.
    constraints = np.concatenate(
        (
            np.column_stack((-values, np.ones(len(values)))),
            np.column_stack((np.eye(num_objectives), np.zeros(num_objectives))),
        )
    )
    equality = np.r_[np.ones(num_objectives), 0.0]
    corners = [*np.eye(num_objectives)]
    for indices in combinations(range(len(constraints)), num_objectives):
        matrix = np.vstack((equality, constraints[list(indices)]))
        try:
            vertex = np.linalg.solve(matrix, np.r_[1.0, np.zeros(num_objectives)])
        except np.linalg.LinAlgError:
            continue
        weight, utility = vertex[:-1], vertex[-1]
        scale = max(1.0, abs(utility), np.abs(values).max())
        if (weight < -1e-7).any() or (values @ weight > utility + 1e-7 * scale).any():
            continue
        weight = np.maximum(weight, 0)
        weight /= weight.sum()
        if not any(np.allclose(weight, old, atol=1e-6) for old in corners):
            corners.append(weight)
    return np.asarray(corners, dtype=np.float32)


def convex_coverage_indices(values):
    """Keep nondominated vectors attaining maximal utility at a simplex corner.

    Equal vectors keep their first representative. Ties between distinct vectors
    are retained when neither dominates the other. Dominated vectors tied at
    a zero-weight boundary do not consume support capacity.
    """
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return np.empty(0, dtype=np.int32)
    unique = []
    for i, value in enumerate(values):
        if not any(np.allclose(value, values[j], atol=1e-6) for j in unique):
            unique.append(i)
    unique = np.asarray(unique)
    candidates = values[unique]
    dominates = np.all(candidates[:, None] >= candidates[None], axis=-1)
    dominates &= np.any(candidates[:, None] > candidates[None], axis=-1)
    unique = unique[~dominates.any(0)]
    candidates = values[unique]
    corners = linear_support_corners(candidates, values.shape[-1])
    utility = candidates @ corners.T
    maximal = np.isclose(utility, utility.max(0), atol=1e-6, rtol=1e-6).any(-1)
    return unique[maximal].astype(np.int32)


class GPILSAgent(Agent):
    """Condition successor features on policy weights and act by GPI."""

    num_objectives: int = pytree_field(static=True)
    agent_network_cfg: DictConfig = pytree_field(static=True)
    num_actions: int = pytree_field(static=True)
    max_support_size: int = pytree_field(default=32, static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)
    discount: float = 0.99
    initial_epsilon: float = 0.1
    huber_delta: float = 1.0

    def init(self, obs_space, action_space, key):
        if not isinstance(action_space, Discrete):
            raise TypeError("GPI-LS requires scalar Discrete actions.")
        q = make_multi_objective_discrete_q_network(
            input_size=obs_space.shape[0] + self.num_objectives,
            action_size=action_space.n,
            hidden_layer_sizes=self.agent_network_cfg.q_hidden_layer_sizes,
            num_objectives=self.num_objectives,
            rngs=nnx.Rngs(key),
        )
        first = jnp.eye(self.num_objectives, dtype=jnp.float32)[0]
        support = jnp.broadcast_to(first, (self.max_support_size, self.num_objectives))
        return AgentState(
            model_state=nnx.unpack(nnx.Dict(q=q, target_q=nnx.clone(q))),
            obs_preprocessor_state=running_statistics.init_state(
                jnp.zeros(obs_space.shape, jnp.float32)
            )
            if self.obs_preprocessor is not None
            else None,
            extra_state=PyTreeDict(
                preferences=first,
                training_preference=first,
                exploration_epsilon=jnp.float32(self.initial_epsilon),
                support_weights=support,
                support_mask=jnp.arange(self.max_support_size) == 0,
                coverage_weights=jnp.zeros_like(support),
                coverage_returns=jnp.zeros_like(support),
                coverage_mask=jnp.zeros(self.max_support_size, jnp.bool_),
                support_iterations=jnp.uint32(0),
                last_support_update=jnp.uint32(0),
            ),
        )

    def evaluate_actions(self, agent_state, sample_batch, key):
        obs = sample_batch.obs.astype(jnp.float32)
        if agent_state.obs_preprocessor_state is not None:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        extra = agent_state.extra_state
        weights = (sample_batch.extras or {}).get("preferences", extra.preferences)
        weights = jnp.broadcast_to(weights, (*obs.shape[:-1], self.num_objectives))
        policy_weights = jnp.broadcast_to(
            extra.support_weights,
            (*obs.shape[:-1], self.max_support_size, self.num_objectives),
        )
        obs = jnp.broadcast_to(
            obs[..., None, :], (*obs.shape[:-1], self.max_support_size, obs.shape[-1])
        )
        values = self.get_models(agent_state, copy=True).q(
            jnp.concatenate((obs, policy_weights), -1)
        )
        return gpi_actions(values, weights, extra.support_mask), PyTreeDict()

    def compute_actions(self, agent_state, sample_batch, key):
        greedy, _ = self.evaluate_actions(agent_state, sample_batch, key)
        explore_key, action_key = jax.random.split(key)
        random = jax.random.randint(action_key, greedy.shape, 0, self.num_actions)
        explore = (
            jax.random.uniform(explore_key, greedy.shape)
            < agent_state.extra_state.exploration_epsilon
        )
        return jnp.where(explore, random, greedy), PyTreeDict()

    def loss(self, agent_state, sample_batch, key):
        """Fit policy-conditioned DDQN vectors, using current and support weights.

        Unlike Envelope Q-learning, Bellman selection maximizes only actions
        for the conditioning policy. GPI is used for behavior and evaluation.
        Truncations bootstrap from original terminal observations.
        """
        extra = agent_state.extra_state
        batch_size = sample_batch.obs.shape[0]
        logits = jnp.where(extra.support_mask, 0.0, -jnp.inf)
        indices = jax.random.categorical(key, logits, shape=(batch_size,))
        weights = jnp.stack(
            (
                jnp.broadcast_to(
                    extra.training_preference, (batch_size, self.num_objectives)
                ),
                extra.support_weights[indices],
            ),
            axis=1,
        )
        obs = jnp.repeat(sample_batch.obs[:, None, :], 2, axis=1)
        next_obs = jnp.repeat(
            sample_batch.extras.env_extras.ori_obs[:, None, :], 2, axis=1
        )
        relabeled = sample_batch.replace(
            obs=obs, extras=PyTreeDict(preferences=weights)
        )
        inputs, _ = preference_inputs(self, agent_state, relabeled)
        next_inputs, _ = preference_inputs(
            self, agent_state, relabeled.replace(obs=next_obs)
        )
        models = self.get_models(agent_state)
        current = jnp.take_along_axis(
            models.q(inputs), sample_batch.actions[:, None, None, None], axis=-2
        ).squeeze(-2)
        next_actions = (
            (models.q(next_inputs) * weights[..., None, :]).sum(-1).argmax(-1)
        )
        next_values = jnp.take_along_axis(
            models.target_q(next_inputs), next_actions[..., None, None], axis=-2
        ).squeeze(-2)
        target = jax.lax.stop_gradient(
            sample_batch.rewards[:, None, :]
            + self.discount
            * (1 - sample_batch.extras.env_extras.termination[:, None, None])
            * next_values
        )
        return PyTreeDict(
            q_loss=optax.huber_loss(current, target, delta=self.huber_delta).mean(),
            q_value=current.mean(),
        )


class GPILSWorkflow(MultiObjectiveOffPolicyWorkflow):
    """GPI-LS with jitted DDQN training and host linear-support stages.

    `step` performs one vectorized rollout and its optimizer updates. `learn`
    additionally calls `update_linear_support` every `training_steps_per_weight`
    steps, evaluates GPI at the current corner weights and selects the task
    offering the largest improvement over the convex coverage set. All support
    and scheduling state is checkpointed inside `AgentState.extra_state`.
    """

    @classmethod
    def name(cls):
        return "GPI-LS"

    @classmethod
    def _build_from_config(cls, config):
        validate_replay_config(config)
        if (
            min(
                config.training_steps_per_weight,
                config.max_support_size,
                config.target_network_update_freq,
                config.exploration_epsilon.decay_updates,
            )
            <= 0
        ):
            raise ValueError(
                "Support, stage and target-update counts must be positive."
            )
        if config.huber_delta <= 0:
            raise ValueError("huber_delta must be positive.")
        if config.num_corner_weights < 0:
            raise ValueError("num_corner_weights must be nonnegative.")
        if not all(
            0 <= value <= 1
            for value in (
                config.exploration_epsilon.start,
                config.exploration_epsilon.end,
            )
        ):
            raise ValueError("Exploration probabilities must be between zero and one.")
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )
        if not isinstance(env.action_space, Discrete):
            raise TypeError("GPI-LS requires scalar Discrete actions.")
        agent = GPILSAgent(
            num_objectives=objective_count(env),
            agent_network_cfg=config.agent_network,
            num_actions=env.action_space.n,
            max_support_size=config.max_support_size,
            discount=config.discount,
            initial_epsilon=config.exploration_epsilon.start,
            huber_delta=config.huber_delta,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
        )
        if config.max_support_size < agent.num_objectives:
            raise ValueError("max_support_size must fit the objective extrema.")
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
            discount=config.discount,
        )
        return cls(env, agent, optimizer, evaluator, replay, config)

    def _setup_agent_and_optimizer(self, key):
        state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        models = self.agent.get_models(state, copy=True)
        return state, nnx.unpack(nnx.Optimizer(models.q, self.optimizer, wrt=nnx.Param))

    def _postsetup_replaybuffer(self, state):
        # Unlike random-utility methods, the first GPI-LS stage starts on one
        # extremum. Later utilities change only at an environment boundary.
        state = state.replace(
            rollout_preferences=jnp.broadcast_to(
                state.agent_state.extra_state.training_preference,
                state.rollout_preferences.shape,
            )
        )
        return super()._postsetup_replaybuffer(state)

    def _rollout(self, state, key, length, random_actions=False):
        """Keep each episode's utility fixed; sample only support preferences."""
        extra = state.agent_state.extra_state
        support = extra.support_weights
        logits = jnp.where(extra.support_mask, 0.0, -jnp.inf)
        random_agent = RandomAgent()
        if random_actions:
            random_state = random_agent.init(
                self.env.obs_space, self.env.action_space, key
            )
        preferences = state.rollout_preferences

        def collect(carry, _):
            env_state, preferences, key = carry
            key, action_key, weight_key = jax.random.split(key, 3)
            batch = SampleBatch(
                obs=env_state.obs, extras=PyTreeDict(preferences=preferences)
            )
            action, _ = (
                random_agent.compute_actions(random_state, batch, action_key)
                if random_actions
                else self.agent.compute_actions(state.agent_state, batch, action_key)
            )
            next_state = self.env.step(env_state, action)
            transition = SampleBatch(
                obs=env_state.obs,
                actions=action,
                rewards=next_state.reward,
                dones=next_state.done,
                extras=PyTreeDict(
                    preferences=preferences,
                    policy_extras=PyTreeDict(),
                    env_extras=PyTreeDict(
                        ori_obs=next_state.info.ori_obs,
                        termination=next_state.info.termination,
                    ),
                ),
            )
            weights = support[
                jax.random.categorical(weight_key, logits, shape=preferences.shape[:-1])
            ]
            preferences = jnp.where(
                next_state.done[..., None].astype(jnp.bool_), weights, preferences
            )
            return (next_state, preferences, key), transition

        axes = jax.sharding.get_abstract_mesh().manual_axes
        (env_state, preferences, _), trajectory = jax.lax.scan(
            collect,
            mark_varying((state.env_state, preferences, key), axes),
            (),
            length=length,
        )
        return trajectory, state.replace(
            env_state=env_state, rollout_preferences=preferences
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
            epsilon = self.config.exploration_epsilon
            agent_state = agent_state.replace(
                model_state=nnx.unpack(models),
                extra_state=agent_state.extra_state.replace(
                    exploration_epsilon=optax.linear_schedule(
                        epsilon.start, epsilon.end, epsilon.decay_updates
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

    def update_linear_support(self, state):
        """Evaluate the support and select the next task on the host.

        This operation is intentionally outside `jax.jit` and `jax.vmap`:
        the number of exact corners depends on observed returns. Network
        evaluation still uses the compiled evaluator, including data parallel
        reduction when enabled. Fixed support capacity never silently drops
        non-dominated policies; increase `max_support_size` if it is exhausted.
        """
        extra = state.agent_state.extra_state
        valid = np.asarray(extra.coverage_mask)
        previous_returns = np.asarray(extra.coverage_returns)[valid]
        previous_weights = np.asarray(extra.coverage_weights)[valid]
        support = np.asarray(extra.support_weights)[np.asarray(extra.support_mask)]
        corners = linear_support_corners(previous_returns, self.agent.num_objectives)
        queries = []
        for weight in np.concatenate((support, corners)):
            if not any(np.allclose(weight, old, atol=1e-6) for old in queries):
                queries.append(weight)
        values = []
        key = state.key
        for weight in queries:
            key, eval_key = jax.random.split(key)
            evaluation_state = state.agent_state.replace(
                extra_state=extra.replace(preferences=jnp.asarray(weight, jnp.float32))
            )
            # Evaluator's policy is pure JAX or batches the host callbacks once.
            value = self._evaluate_support(evaluation_state, eval_key)
            values.append(np.asarray(value))
        returns = np.concatenate((previous_returns, np.asarray(values)))
        weights = np.concatenate((previous_weights, np.asarray(queries)))
        selected = convex_coverage_indices(returns)
        coverage_returns, coverage_weights = returns[selected], weights[selected]
        if len(selected) > self.agent.max_support_size:
            raise ValueError(
                "GPI-LS coverage exceeded max_support_size; increase the configured capacity."
            )
        next_corners = linear_support_corners(
            coverage_returns, self.agent.num_objectives
        )
        # The initial stages train every extremum, then use the GPI improvement
        # over the current coverage surface as the reference implementation does.
        iteration = int(extra.support_iterations) + 1
        if iteration < self.agent.num_objectives:
            next_weight = np.eye(self.agent.num_objectives, dtype=np.float32)[iteration]
        else:
            expanded = []
            for weight in next_corners:
                key, eval_key = jax.random.split(key)
                value = self._evaluate_support(
                    state.agent_state.replace(
                        extra_state=extra.replace(preferences=jnp.asarray(weight))
                    ),
                    eval_key,
                )
                expanded.append(np.asarray(value))
            gains = (np.asarray(expanded) @ next_corners.T).max(0) - (
                coverage_returns @ next_corners.T
            ).max(0)
            # Rotate exact ties deterministically, so zero gains do not starve
            # all but the first corner and checkpoint continuation is exact.
            ties = np.flatnonzero(np.isclose(gains, gains.max(), atol=1e-6))
            next_weight = next_corners[ties[iteration % len(ties)]]
        candidates = [next_weight]
        for weight in np.concatenate(
            (coverage_weights, next_corners[: self.config.num_corner_weights])
        ):
            if not any(np.allclose(weight, old, atol=1e-6) for old in candidates):
                candidates.append(weight)
        if len(candidates) > self.agent.max_support_size:
            raise ValueError(
                "GPI-LS support exceeded max_support_size; increase the configured capacity."
            )
        support_weights = np.zeros_like(np.asarray(extra.support_weights))
        support_weights[: len(candidates)] = candidates
        padded_returns = np.zeros_like(np.asarray(extra.coverage_returns))
        padded_weights = np.zeros_like(np.asarray(extra.coverage_weights))
        padded_returns[: len(selected)] = coverage_returns
        padded_weights[: len(selected)] = coverage_weights
        replacements = dict(
            training_preference=next_weight,
            preferences=next_weight,
            support_weights=support_weights,
            support_mask=np.arange(len(support_weights)) < len(candidates),
            coverage_returns=padded_returns,
            coverage_weights=padded_weights,
            coverage_mask=np.arange(len(support_weights)) < len(selected),
            support_iterations=np.uint32(iteration),
            last_support_update=np.asarray(state.metrics.iterations),
        )
        replacements = {
            name: jax.device_put(
                jnp.asarray(value, dtype=extra[name].dtype), extra[name].sharding
            )
            for name, value in replacements.items()
        }
        return state.replace(
            key=key,
            agent_state=state.agent_state.replace(
                extra_state=extra.replace(**replacements)
            ),
        )

    def learn(self, state):
        """Train, refresh linear support at stage boundaries, record and save."""
        stride = self.config.rollout_length * self.config.num_envs * self.num_devices
        remaining = max(
            0, self.config.total_timesteps - int(state.metrics.sampled_timesteps)
        )
        num_steps = (remaining + stride - 1) // stride
        for index in range(num_steps):
            train, state = self.step(state)
            extra = state.agent_state.extra_state
            if (
                int(state.metrics.iterations - extra.last_support_update)
                >= self.config.training_steps_per_weight
            ):
                state = self.update_linear_support(state)
            iteration = int(state.metrics.iterations)
            self.recorder.write(train.to_local_dict(), iteration)
            self.recorder.write(state.metrics.to_local_dict(), iteration)
            final = index == num_steps - 1
            if iteration % self.config.eval_interval == 0 or final:
                evaluation, state = self.evaluate(state)
                self.recorder.write(
                    {
                        f"eval/{key}": value
                        for key, value in evaluation.to_local_dict().items()
                    },
                    iteration,
                )
            saved = (
                state
                if self.config.save_replay_buffer
                else state.replace(replay_buffer_state=None)
            )
            self.checkpoint_manager.save(iteration, saved, force=final)
        return state

    def _evaluate_support(self, agent_state, key):
        if self.is_multi_device:
            key = jax.random.fold_in(key, jax.lax.axis_index(self.dp_axis_name))
        result = self.evaluator.evaluate(agent_state, key, self.config.eval_episodes)
        return pmean(result.episode_returns.mean(0), self.dp_axis_name)

    @classmethod
    def enable_shmap(cls):
        super().enable_shmap()
        cls._evaluate_support = jax.shard_map(
            cls._evaluate_support,
            in_specs=(None, jax.P(), jax.P()),
            out_specs=jax.P(),
        )

    @classmethod
    def enable_jit(cls):
        super().enable_jit()
        cls._evaluate_support = jax.jit(cls._evaluate_support, static_argnums=(0,))
