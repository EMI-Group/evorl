"""Decomposition directions, PSA adaptation and policy archives for MORL/D."""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from evorl.ec.optimizers.nsga2 import crowding_distance
from evorl.types import PyTreeDict
from evorl.utils.jax_utils import rng_fold_in_axis


def reset_decomposition_env(workflow, key):
    """Reset one subproblem's environment batch with its normal DP layout."""
    if not workflow.is_multi_device:
        return workflow.env.reset(key)

    def reset(key):
        key = rng_fold_in_axis(key, workflow.dp_axis_name)
        return workflow.env.reset(key)

    return jax.shard_map(
        reset, in_specs=jax.P(), out_specs=jax.P(workflow.dp_axis_name)
    )(key)


def decomposition_weights(num_objectives, pop_size):
    """Create deterministic simplex-lattice directions, retaining spread points.

    Two objectives use evenly spaced directions. More objectives use a simplex
    lattice followed by farthest-point selection, without a dependency on pymoo.
    """
    if num_objectives == 1:
        return np.ones((pop_size, 1), np.float32)
    if num_objectives == 2:
        values = np.linspace(0, 1, pop_size, dtype=np.float32)
        return np.stack((values, 1 - values), axis=-1)
    divisions = 1
    while math.comb(divisions + num_objectives - 1, num_objectives - 1) < pop_size:
        divisions += 1
    directions = []
    for cuts in itertools.combinations(
        range(divisions + num_objectives - 1), num_objectives - 1
    ):
        limits = (-1, *cuts, divisions + num_objectives - 1)
        directions.append(np.diff(limits) - 1)
    candidates = np.asarray(directions, np.float32) / divisions
    selected = [0]
    distances = np.square(candidates - candidates[0]).sum(-1)
    for _ in range(pop_size - 1):
        distances[selected] = -1
        index = int(distances.argmax())
        selected.append(index)
        distances = np.minimum(
            distances, np.square(candidates - candidates[index]).sum(-1)
        )
    return candidates[selected]


def nearest_decomposition_neighbors(preferences, count):
    """Return the nearest other subproblems in squared preference distance."""
    distance = jnp.square(preferences[:, None] - preferences[None]).sum(-1)
    distance = jnp.where(jnp.eye(len(preferences), dtype=jnp.bool_), jnp.inf, distance)
    return jnp.argsort(distance, axis=-1, stable=True)[:, :count]


def psa_preferences(preferences, returns, archive_returns, archive_valid, delta):
    """Apply the reference PSA update relative to the nearest distinct return.

    Better coordinates gain a factor `1 + delta`; worse coordinates lose that
    factor. A point with no distinct archive neighbor keeps its preference.
    """
    distance = jnp.square(returns[:, None] - archive_returns[None]).sum(-1)
    eligible = archive_valid[None] & (distance > 0.01)
    nearest = jnp.argmin(jnp.where(eligible, distance, jnp.inf), axis=-1)
    reference = archive_returns[nearest]
    factors = jnp.where(returns >= reference, 1 + delta, 1 / (1 + delta))
    updated = preferences * factors
    updated /= updated.sum(-1, keepdims=True)
    return jnp.where(eligible.any(-1, keepdims=True), updated, preferences)


def update_decomposition_archive(archive, returns, agent_states, preferences):
    """Keep distinct nondominated returns and their matching policy snapshots.

    A configured capacity bounds model storage. Crowding distance removes the
    least diverse points when the front exceeds that capacity; ties retain the
    earlier snapshot. Invalid slots never participate in dominance or equality.
    """
    capacity = archive.returns.shape[0]
    values = jnp.concatenate((archive.returns, returns))
    valid = jnp.concatenate((archive.valid, jnp.ones(returns.shape[0], jnp.bool_)))
    all_weights = jnp.concatenate((archive.preferences, preferences))
    all_states = jax.tree.map(
        lambda old, new: jnp.concatenate((old, new)), archive.agent_states, agent_states
    )
    better = jnp.all(values[:, None] >= values[None], axis=-1)
    better &= jnp.any(values[:, None] > values[None], axis=-1)
    dominated = (better & valid[:, None]).any(axis=0)
    equal = jnp.all(values[:, None] == values[None], axis=-1)
    earlier = jnp.arange(len(values))[:, None] < jnp.arange(len(values))[None]
    duplicate = (equal & earlier & valid[:, None]).any(axis=0)
    keep = valid & ~dominated & ~duplicate
    ranks = jnp.where(keep, 0, 1)
    distance = crowding_distance(values, ranks)
    order = jnp.lexsort((jnp.arange(len(values)), -distance, ~keep))[:capacity]
    return PyTreeDict(
        returns=values[order],
        valid=keep[order],
        preferences=all_weights[order],
        agent_states=jax.tree.map(lambda x: x[order], all_states),
    )
