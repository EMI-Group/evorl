"""Prediction-guided MORL (Xu et al., ICML 2020).

Warmup and evolutionary policy training follow MORL-Baselines. The population,
performance buffer, predictor observations and Pareto policy archive are explicit
checkpoint state. The paper's Pareto-analysis/interpolation postprocessing is
not included, matching the reference implementation's documented scope.
"""

import copy
import math

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import least_squares

from evorl.envs import Box
from evorl.metrics import TrainMetric
from evorl.recorders import add_prefix
from evorl.types import State
from evorl.workflows import Workflow

from .common import MultiObjectiveEvaluateMetric, MultiObjectiveWorkflowMetric
from .moppo import MOPPOWorkflow


def generate_weights(divisions, dimensions):
    """Generate a simplex lattice without a Cartesian-product allocation."""
    if divisions < 1 or dimensions < 1:
        raise ValueError("Weight divisions and dimensions must be positive")

    def compositions(total, size):
        if size == 1:
            yield (total,)
        else:
            for first in range(total + 1):
                for rest in compositions(total - first, size - 1):
                    yield (first, *rest)

    return np.asarray(list(compositions(divisions, dimensions)), np.float32) / divisions


def pareto_indices(values):
    """Return unique, non-dominated indices for maximized reward vectors."""
    indices = []
    for i, value in enumerate(values):
        if any(np.all(other >= value) and np.any(other > value) for other in values):
            continue
        if any(np.array_equal(value, values[j]) for j in indices):
            continue
        indices.append(i)
    return np.asarray(indices, np.int32)


def hypervolume(values, reference):
    """Compute exact maximization hypervolume in two or three dimensions."""
    shifted = np.asarray(values, np.float64).reshape(-1, len(reference)) - reference
    shifted = shifted[np.all(shifted > 0, axis=1)]

    def volume(points):
        if len(points) == 0:
            return 0.0
        if points.shape[1] == 1:
            return float(points[:, 0].max())
        result, previous = 0.0, 0.0
        for coordinate in np.unique(points[:, 0]):
            result += (coordinate - previous) * volume(
                points[points[:, 0] >= coordinate, 1:]
            )
            previous = coordinate
        return result

    return volume(shifted)


def sparsity(values):
    """Use the reference's mean squared gaps along sorted objective coordinates."""
    values = np.asarray(values)
    if len(values) < 2:
        return 0.0
    return float(
        np.square(np.diff(np.sort(values, axis=0), axis=0)).sum() / (len(values) - 1)
    )


def predict_performance(
    weights, before, after, current, candidate, sigma=0.03, neighborhood=0.1
):
    """Fit the reference's local, bounded hyperbolic performance-delta model.

    The tanh form is algebraically equal to its exponential quotient and avoids
    overflow. Neighborhood expansion uses a finite sorted-distance threshold;
    sparse early histories fall back to a weighted mean until four observations
    exist. Zero-valued objective coordinates remain valid model inputs.
    """
    weights, before, after = map(np.asarray, (weights, before, after))
    candidate = np.asarray(candidate)
    if len(weights) == 0:
        return np.broadcast_to(np.asarray(current, np.float64), candidate.shape).copy()
    scale = np.maximum(np.abs(current), 1.0)
    distances = np.max(np.abs(before - current) / scale, axis=1)
    threshold = max(
        neighborhood,
        float(
            np.partition(distances, min(3, len(distances) - 1))[
                min(3, len(distances) - 1)
            ]
        )
        + 1e-12,
    )
    selected = distances <= threshold
    weights, after, delta = (
        weights[selected],
        after[selected],
        (after - before)[selected],
    )
    local_sigma = max(sigma, threshold * sigma / neighborhood)
    importance = np.exp(
        -0.5
        * np.square(np.linalg.norm((after - current) / scale, axis=1) / local_sigma)
    )
    importance = np.maximum(importance, 1e-12)
    prediction = []
    for objective in range(len(current)):
        y, x = delta[:, objective], weights[:, objective]
        if len(y) < 4 or np.ptp(x) < 1e-8:
            prediction.append(
                np.full(candidate.shape[:-1], np.average(y, weights=importance))
            )
            continue
        amplitude = float(np.clip(np.ptp(y), 1.0, 500.0))

        def residual(params, x=x, y=y):
            a, slope, offset, bias = params
            return (a * np.tanh(0.5 * slope * (x - offset)) + bias - y) * importance

        fit = least_squares(
            residual,
            [min(1.0, amplitude), 1.0, 0.0, float(np.clip(y.mean(), -499, 499))],
            bounds=([0.0, 0.1, -5.0, -500.0], [amplitude, 20.0, 5.0, 500.0]),
            loss="soft_l1",
            f_scale=20.0,
            max_nfev=200,
        )
        a, slope, offset, bias = fit.x
        prediction.append(
            a * np.tanh(0.5 * slope * (candidate[..., objective] - offset)) + bias
        )
    return np.asarray(current) + np.stack(prediction, axis=-1)


def _take(tree, index):
    return jax.tree.map(lambda x: x[index], tree)


def _put(tree, index, value):
    return jax.tree.map(lambda x, y: x.at[index].set(y), tree, value)


def _train_population(workflow, population):
    return jax.vmap(workflow.inner.step)(population)


def _evaluate_population(workflow, population):
    return jax.vmap(workflow.inner.evaluate)(population)


def _evaluate_archive(workflow, agents, keys):
    return jax.vmap(
        lambda agent, key: workflow.evaluator.evaluate(
            agent, key, workflow.config.eval_episodes
        )
    )(agents, keys)


class PGMORLWorkflow(Workflow):
    """Select policy/weight tasks on the host and train JIT-compiled PPO workers.

    Supports two or three maximized objectives and continuous actions. Task
    selection deliberately stays outside JIT because it fits local regression
    models. Population training and evaluation are JIT-compiled and vmapped.
    """

    def __init__(self, inner, config, jit=True):
        super().__init__(config)
        self.inner = inner
        self.jit = jit
        self.agent = inner.agent
        self.env = inner.env
        self.evaluator = inner.evaluator
        self.directions = generate_weights(
            config.performance_buffer_divisions, inner.agent.num_objectives
        )
        self.directions /= np.linalg.norm(self.directions, axis=1, keepdims=True)
        self.candidate_weights = generate_weights(
            config.weight_divisions, inner.agent.num_objectives
        )

    @classmethod
    def name(cls):
        return "PGMORL"

    @classmethod
    def build_from_config(cls, config, multi_device=False, jit=True):
        if multi_device:
            raise ValueError("PGMORL currently supports a single device")
        config = copy.deepcopy(config)
        if (
            min(
                config.pop_size,
                config.warmup_iterations,
                config.evolutionary_iterations,
                config.archive_capacity,
                config.performance_buffer_size,
                config.weight_divisions,
                config.performance_buffer_divisions,
            )
            <= 0
        ):
            raise ValueError(
                "Population, training, buffer and weight counts must be positive"
            )
        if config.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        inner_config = copy.deepcopy(config)
        OmegaConf.set_struct(inner_config, False)
        inner_config.checkpoint.enable = False
        inner_cls = type("PGMORLInnerPPO", (MOPPOWorkflow,), {})
        inner = inner_cls.build_from_config(inner_config, jit=jit)
        objectives = inner.agent.num_objectives
        if objectives not in (2, 3) or not isinstance(inner.env.action_space, Box):
            inner.close()
            raise ValueError(
                "PGMORL requires continuous actions and two or three objectives"
            )
        reference = np.asarray(config.reference_point)
        if reference.shape != (objectives,) or not np.isfinite(reference).all():
            inner.close()
            raise ValueError(
                "reference_point must contain one finite value per objective"
            )
        if config.predictor_sigma <= 0 or config.predictor_neighborhood <= 0:
            inner.close()
            raise ValueError("Predictor scales must be positive")
        OmegaConf.set_readonly(config, True)
        return cls(inner, config, jit=jit)

    def setup(self, key):
        key, init_key = jax.random.split(key)
        setup_fn = jax.vmap(self.inner.setup)
        if self.jit:
            setup_fn = jax.jit(setup_fn)
        population = setup_fn(jax.random.split(init_key, self.config.pop_size))
        lattice = generate_weights(
            max(1, self.config.pop_size - 1), self.agent.num_objectives
        )
        # Farthest-point sampling covers all objective extremes before filling
        # the simplex interior, including in the three-objective case.
        selected = [0]
        distance = np.square(lattice - lattice[0]).sum(-1)
        for _ in range(self.config.pop_size - 1):
            distance[selected] = -1
            index = int(distance.argmax())
            selected.append(index)
            distance = np.minimum(distance, np.square(lattice - lattice[index]).sum(-1))
        weights = lattice[selected]
        population = population.replace(
            agent_state=population.agent_state.replace(
                extra_state=population.agent_state.extra_state.replace(
                    preferences=jnp.asarray(weights)
                )
            )
        )
        evaluate_fn = (
            jax.jit(_evaluate_population, static_argnums=0)
            if self.jit
            else _evaluate_population
        )
        evaluation, population = evaluate_fn(self, population)
        capacity = len(self.directions) * self.config.performance_buffer_size
        template = jax.tree.map(
            lambda x: jnp.repeat(x[:1], capacity, axis=0), population.agent_state
        )
        optimizer = jax.tree.map(
            lambda x: jnp.repeat(x[:1], capacity, axis=0), population.opt_state
        )
        archive = jax.tree.map(
            lambda x: jnp.repeat(x[:1], self.config.archive_capacity, axis=0),
            population.agent_state,
        )
        history_capacity = (
            math.ceil(
                self.config.total_timesteps
                / (self.config.num_envs * self.config.rollout_length)
            )
            + self.config.pop_size
        )
        shape = (history_capacity, self.agent.num_objectives)
        state = State(
            key=key,
            population=population,
            performances=evaluation.episode_returns,
            metrics=MultiObjectiveWorkflowMetric(),
            pool_agents=template,
            pool_optimizers=optimizer,
            pool_returns=jnp.zeros((capacity, self.agent.num_objectives)),
            pool_valid=jnp.zeros(capacity, jnp.bool_),
            archive_agents=archive,
            archive_returns=jnp.zeros(
                (self.config.archive_capacity, self.agent.num_objectives)
            ),
            archive_count=jnp.int32(0),
            history_weights=jnp.zeros(shape),
            history_before=jnp.zeros(shape),
            history_after=jnp.zeros(shape),
            history_count=jnp.int32(0),
        )
        return self._retain_population(state)

    def _retain_population(self, state):
        """Retain radial-buffer snapshots and non-dominated evaluated policies."""
        returns = np.asarray(state.pool_returns).copy()
        valid = np.asarray(state.pool_valid).copy()
        agents, optimizers = state.pool_agents, state.pool_optimizers
        performances = np.asarray(state.performances)
        reference = np.asarray(self.config.reference_point)
        for index, value in enumerate(performances):
            shifted = np.maximum(value - reference, 0)
            direction = int(
                np.argmax(
                    self.directions @ (shifted / max(np.linalg.norm(shifted), 1e-12))
                )
            )
            first = direction * self.config.performance_buffer_size
            slots = np.arange(first, first + self.config.performance_buffer_size)
            free = slots[~valid[slots]]
            if len(free):
                slot = int(free[0])
            else:
                norms = np.linalg.norm(
                    np.maximum(returns[slots] - reference, 0), axis=1
                )
                slot = int(slots[np.argmin(norms)])
                if np.linalg.norm(shifted) <= norms.min():
                    continue
            returns[slot], valid[slot] = value, True
            agents = _put(agents, slot, _take(state.population.agent_state, index))
            optimizers = _put(
                optimizers, slot, _take(state.population.opt_state, index)
            )
        count = int(state.archive_count)
        combined = np.concatenate(
            (np.asarray(state.archive_returns)[:count], performances)
        )
        combined_agents = jax.tree.map(
            lambda a, b: jnp.concatenate((a[:count], b)),
            state.archive_agents,
            state.population.agent_state,
        )
        indices = pareto_indices(combined)
        if len(indices) > self.config.archive_capacity:
            # Keep sparse front representatives, preserving coordinate extremes.
            scores = np.zeros(len(indices))
            front = combined[indices]
            for objective in range(self.agent.num_objectives):
                order = np.argsort(front[:, objective])
                scores[order[[0, -1]]] = np.inf
                scale = max(float(np.ptp(front[:, objective])), 1e-12)
                scores[order[1:-1]] += (
                    front[order[2:], objective] - front[order[:-2], objective]
                ) / scale
            indices = indices[
                np.argsort(-scores, kind="stable")[: self.config.archive_capacity]
            ]
        archive_returns = state.archive_returns.at[: len(indices)].set(
            combined[indices]
        )
        archive_agents = jax.tree.map(
            lambda a, b: a.at[: len(indices)].set(b[indices]),
            state.archive_agents,
            combined_agents,
        )
        return state.replace(
            pool_agents=agents,
            pool_optimizers=optimizers,
            pool_returns=jnp.asarray(returns),
            pool_valid=jnp.asarray(valid),
            archive_agents=archive_agents,
            archive_returns=archive_returns,
            archive_count=jnp.int32(len(indices)),
        )

    def _select_tasks(self, state):
        count = int(state.history_count)
        previous = np.asarray(state.history_before)[:count]
        following = np.asarray(state.history_after)[:count]
        used_weights = np.asarray(state.history_weights)[:count]
        pool = np.asarray(state.pool_returns)
        valid = np.flatnonzero(np.asarray(state.pool_valid))
        front = list(np.asarray(state.archive_returns)[: int(state.archive_count)])
        predictions = []
        for policy in valid:
            predicted = predict_performance(
                used_weights,
                previous,
                following,
                pool[policy],
                self.candidate_weights,
                self.config.predictor_sigma,
                self.config.predictor_neighborhood,
            )
            predictions.extend(
                (int(policy), weight, value)
                for weight, value in zip(self.candidate_weights, predicted, strict=True)
            )
        selected, remaining = [], list(range(len(predictions)))
        for _ in range(self.config.pop_size):
            if not remaining:
                remaining = list(range(len(predictions)))
            scores = []
            for index in remaining:
                estimated = np.asarray([*front, predictions[index][2]])
                scores.append(
                    hypervolume(estimated, self.config.reference_point)
                    + self.config.sparsity_coef * sparsity(estimated)
                )
            best = remaining.pop(int(np.argmax(scores)))
            policy, weight, predicted = predictions[best]
            selected.append((policy, weight))
            front.append(predicted)
        indices = jnp.asarray([p for p, _ in selected])
        weights = jnp.asarray(np.asarray([w for _, w in selected]), jnp.float32)
        population = state.population.replace(
            agent_state=_take(state.pool_agents, indices),
            opt_state=_take(state.pool_optimizers, indices),
        )
        population = population.replace(
            agent_state=population.agent_state.replace(
                extra_state=population.agent_state.extra_state.replace(
                    preferences=weights
                )
            )
        )
        # Environments, root keys and counters stay with the destination workers.
        return state.replace(
            population=population, performances=state.pool_returns[indices]
        )

    def step(self, state):
        remaining = self.config.total_timesteps - int(state.metrics.sampled_timesteps)
        if remaining <= 0:
            return TrainMetric(), state
        if int(state.metrics.iterations) > 0:
            state = self._select_tasks(state)
        before = state.performances
        iterations = (
            self.config.warmup_iterations
            if int(state.metrics.iterations) == 0
            else self.config.evolutionary_iterations
        )
        quantum = self.config.num_envs * self.config.rollout_length
        updates = min(iterations * self.config.pop_size, math.ceil(remaining / quantum))
        if getattr(self.env, "uses_host_callback", False):
            # A native pool owns the whole population batch and cannot step a
            # subset. Finish that batch and count every collected transition.
            updates = math.ceil(updates / self.config.pop_size) * self.config.pop_size
        full, tail = divmod(updates, self.config.pop_size)
        population = state.population
        losses = []
        train_fn = (
            jax.jit(_train_population, static_argnums=0)
            if self.jit
            else _train_population
        )
        for _ in range(full):
            metric, population = train_fn(self, population)
            losses.extend(np.asarray(metric.loss).tolist())
        for worker in range(tail):
            metric, updated = self.inner.step(_take(population, worker))
            population = _put(population, worker, updated)
            losses.append(float(metric.loss))
        evaluate_fn = (
            jax.jit(_evaluate_population, static_argnums=0)
            if self.jit
            else _evaluate_population
        )
        evaluation, population = evaluate_fn(self, population)
        trained = (
            np.ones(self.config.pop_size, bool)
            if full
            else np.arange(self.config.pop_size) < tail
        )
        history_count = int(state.history_count)
        trained_indices = jnp.asarray(np.flatnonzero(trained))
        n = len(trained_indices)
        state = state.replace(
            population=population,
            performances=evaluation.episode_returns,
            history_weights=state.history_weights.at[
                history_count : history_count + n
            ].set(population.agent_state.extra_state.preferences[trained_indices]),
            history_before=state.history_before.at[
                history_count : history_count + n
            ].set(before[trained_indices]),
            history_after=state.history_after.at[history_count : history_count + n].set(
                evaluation.episode_returns[trained_indices]
            ),
            history_count=jnp.int32(history_count + n),
            metrics=state.metrics.replace(
                sampled_timesteps=state.metrics.sampled_timesteps
                + jnp.uint32(updates * quantum),
                sampled_episodes=population.metrics.sampled_episodes.sum(),
                training_updates=state.metrics.training_updates
                + jnp.uint32(
                    updates
                    * self.config.reuse_rollout_epochs
                    * quantum
                    // self.config.minibatch_size
                ),
                iterations=state.metrics.iterations + 1,
            ),
        )
        return TrainMetric(loss=jnp.float32(np.mean(losses))), self._retain_population(
            state
        )

    def evaluate(self, state):
        count = int(state.archive_count)
        key, eval_key = jax.random.split(state.key)
        agents = jax.tree.map(lambda x: x[:count], state.archive_agents)
        preferences = agents.extra_state.preferences
        evaluate_fn = (
            jax.jit(_evaluate_archive, static_argnums=0)
            if self.jit
            else _evaluate_archive
        )
        evaluation = evaluate_fn(self, agents, jax.random.split(eval_key, count))
        returns = evaluation.episode_returns.mean(axis=1)
        return MultiObjectiveEvaluateMetric(
            preferences=preferences,
            episode_returns=returns,
            scalarized_returns=(preferences * returns).sum(-1),
            episode_lengths=evaluation.episode_lengths.mean(axis=1),
        ), state.replace(key=key)

    def learn(self, state):
        while int(state.metrics.sampled_timesteps) < self.config.total_timesteps:
            metrics, state = self.step(state)
            iteration = int(state.metrics.iterations)
            self.recorder.write(state.metrics.to_local_dict(), iteration)
            self.recorder.write(metrics.to_local_dict(), iteration)
            complete = (
                int(state.metrics.sampled_timesteps) >= self.config.total_timesteps
            )
            if iteration % self.config.eval_interval == 0 or complete:
                evaluation, state = self.evaluate(state)
                self.recorder.write(
                    add_prefix(evaluation.to_local_dict(), "eval"), iteration
                )
            self.checkpoint_manager.save(iteration, state, force=complete)
        return state

    def close(self):
        try:
            self.inner.close()
        finally:
            super().close()
