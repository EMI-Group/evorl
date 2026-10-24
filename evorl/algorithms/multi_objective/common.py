"""Preference sampling, vector replay and evaluation shared by MORL workflows."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from evorl.agent import RandomAgent
from evorl.algorithms.offpolicy_utils import OffPolicyWorkflowTemplate, clean_trajectory
from evorl.distributed import DP_AXIS_NAME, pmean, psum
from evorl.distributed.sharding import mark_varying
from evorl.envs import Box
from evorl.metrics import MetricBase, WorkflowMetric, metric_field
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict, State
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory
from evorl.workflows.rl_workflow import _shard_map_replay_buffer_context


def sample_preferences(key, shape, num_objectives):
    """Sample nonnegative, L1-normalized Gaussian weights, as in the references."""
    if num_objectives == 1:
        return jnp.ones((*shape, 1), jnp.float32)
    weights = jnp.abs(
        jax.random.normal(key, (*shape, num_objectives), dtype=jnp.float32)
    )
    return weights / jnp.maximum(
        weights.sum(-1, keepdims=True), jnp.finfo(weights.dtype).tiny
    )


def objective_count(env):
    """Validate a single-agent vector reward and flat numeric observations."""
    if not isinstance(env.obs_space, Box) or len(env.obs_space.shape) != 1:
        raise TypeError(
            "MORL workflows currently require one-dimensional Box observations."
        )
    reward = jax.eval_shape(env.reset, jax.random.key(0)).reward
    if not isinstance(reward, jax.ShapeDtypeStruct) or reward.ndim != 2:
        raise ValueError(
            "MORL requires vector rewards of shape (num_envs, num_objectives), not scalar or agent-keyed rewards."
        )
    if reward.shape[0] != env.num_envs or reward.shape[1] < 1:
        raise ValueError("Invalid vector reward shape.")
    return reward.shape[1]


def preference_inputs(agent, agent_state, sample_batch):
    """Normalize observations and append explicit or default preference weights.

    Training batches carry `extras.preferences`. Direct policy calls use
    `agent_state.extra_state.preferences`, which can be replaced for inference.
    Preference coordinates are never observation-normalized.
    """
    obs = sample_batch.obs.astype(jnp.float32)
    if agent_state.obs_preprocessor_state is not None:
        obs = agent.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
    preferences = (sample_batch.extras or {}).get(
        "preferences", agent_state.extra_state.preferences
    )
    preferences = jnp.broadcast_to(preferences, (*obs.shape[:-1], agent.num_objectives))
    return jnp.concatenate((obs, preferences), axis=-1), preferences


class MultiObjectiveWorkflowMetric(WorkflowMetric):
    """Count optimizer updates separately from sampled environment transitions."""

    training_updates: jax.Array = metric_field(default_factory=lambda: jnp.uint32(0))


class MultiObjectiveEvaluateMetric(MetricBase):
    """Keep every evaluation preference and reward coordinate visible.

    `episode_returns` has shape `(preferences, objectives)`; scalarized returns
    and episode lengths have shape `(preferences,)`. Only episode/device axes
    are averaged, never the objective or preference axis.
    """

    preferences: jax.Array
    episode_returns: jax.Array = metric_field(reduce_fn=pmean)
    scalarized_returns: jax.Array = metric_field(reduce_fn=pmean)
    episode_lengths: jax.Array = metric_field(reduce_fn=pmean)


class MultiObjectiveOffPolicyWorkflow(OffPolicyWorkflowTemplate):
    """Keep rollout preferences alongside the environment, outside model state.

    Models and optimizers are replicated in data-parallel training. Environment
    states, rollout preferences and replay rings are sharded together. Each
    worker collects and samples its own transitions; only explicit gradient,
    observation-statistic and metric reductions communicate.
    """

    resample_preferences_every_step = False

    def __init__(self, env, agent, optimizer, evaluator, replay_buffer, config):
        super().__init__(env, agent, optimizer, evaluator, replay_buffer, config)
        weights = config.eval_preferences
        if weights is None:
            weights = np.concatenate(
                (
                    np.eye(agent.num_objectives),
                    np.ones((1, agent.num_objectives)) / agent.num_objectives,
                )
            )
            if agent.num_objectives == 1:
                weights = weights[:1]
        weights = np.asarray(weights, dtype=np.float32)
        if (
            weights.ndim != 2
            or weights.shape[0] == 0
            or weights.shape[1] != agent.num_objectives
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.allclose(weights.sum(-1), 1, atol=1e-6)
        ):
            raise ValueError(
                "eval_preferences must be nonnegative vectors summing to one, with one entry per objective."
            )
        self.eval_preferences = tuple(map(tuple, weights.tolist()))

    def setup(self, key):
        key, agent_key, env_key, buffer_key = jax.random.split(key, 4)
        agent_state, opt_state = self._setup_agent_and_optimizer(agent_key)
        metrics = MultiObjectiveWorkflowMetric()

        def reset(env_key):
            if self.is_multi_device:
                env_key = jax.random.fold_in(
                    env_key, jax.lax.axis_index(self.dp_axis_name)
                )
            reset_key, weight_key = jax.random.split(env_key)
            return self.env.reset(reset_key), sample_preferences(
                weight_key, (self.config.num_envs,), self.agent.num_objectives
            )

        if self.is_multi_device:
            key, env_key, buffer_key, agent_state, opt_state, metrics = jax.device_put(
                (key, env_key, buffer_key, agent_state, opt_state, metrics), jax.P()
            )
            env_state, preferences = jax.shard_map(
                reset,
                in_specs=jax.P(),
                out_specs=(jax.P(self.dp_axis_name), jax.P(self.dp_axis_name)),
            )(env_key)

            def init_buffer(buffer_key):
                buffer = self._setup_replaybuffer(buffer_key)
                return buffer.replace(
                    current_index=buffer.current_index[None],
                    buffer_size=buffer.buffer_size[None],
                )

            buffer_state = jax.shard_map(
                init_buffer, in_specs=jax.P(), out_specs=jax.P(self.dp_axis_name)
            )(buffer_key)
        else:
            env_state, preferences = reset(env_key)
            buffer_state = self._setup_replaybuffer(buffer_key)
        return self._postsetup_replaybuffer(
            State(
                key=key,
                agent_state=agent_state,
                opt_state=opt_state,
                metrics=metrics,
                env_state=env_state,
                replay_buffer_state=buffer_state,
                rollout_preferences=preferences,
            )
        )

    def _setup_replaybuffer(self, key):
        env = jax.eval_shape(self.env.reset, key)
        zeros = lambda x: jnp.zeros(x.shape[1:], x.dtype)
        spec = SampleBatch(
            obs=zeros(env.obs),
            actions=self.env.action_space.sample(key),
            rewards=zeros(env.reward),
            extras=PyTreeDict(
                preferences=jnp.zeros(self.agent.num_objectives, jnp.float32),
                policy_extras=PyTreeDict(),
                env_extras=PyTreeDict(
                    ori_obs=zeros(env.obs), termination=zeros(env.info.termination)
                ),
            ),
        )
        return self.replay_buffer.init(spec)

    def _rollout(self, state, key, length, random_actions=False):
        """Collect original next observations and the preference used for each action."""
        random_agent = RandomAgent()
        if random_actions:
            random_state = random_agent.init(
                self.env.obs_space, self.env.action_space, key
            )

        def step(carry, _):
            env_state, preferences, key = carry
            key, action_key, weight_key = jax.random.split(key, 3)
            batch = SampleBatch(
                obs=env_state.obs, extras=PyTreeDict(preferences=preferences)
            )
            if random_actions:
                action, _ = random_agent.compute_actions(
                    random_state, batch, action_key
                )
            else:
                action, _ = self.agent.compute_actions(
                    state.agent_state, batch, action_key
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
            new_weights = sample_preferences(
                weight_key, preferences.shape[:-1], self.agent.num_objectives
            )
            replace = (
                jnp.ones_like(next_state.done, jnp.bool_)
                if self.resample_preferences_every_step
                else next_state.done.astype(jnp.bool_)
            )
            preferences = jnp.where(replace[..., None], new_weights, preferences)
            return (next_state, preferences, key), transition

        axes = jax.sharding.get_abstract_mesh().manual_axes
        (env_state, preferences, _), trajectory = jax.lax.scan(
            step,
            mark_varying((state.env_state, state.rollout_preferences, key), axes),
            (),
            length=length,
        )
        return trajectory, state.replace(
            env_state=env_state, rollout_preferences=preferences
        )

    def _postsetup_replaybuffer(self, state):
        training_preferences = state.rollout_preferences
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
        key, random_key, policy_key = jax.random.split(state.key, 3)
        if self.is_multi_device:
            rank = jax.lax.axis_index(self.dp_axis_name)
            random_key, policy_key = (
                jax.random.fold_in(k, rank) for k in (random_key, policy_key)
            )
        state = state.replace(key=key)
        for length, rollout_key, random in (
            (random_length, random_key, True),
            (policy_length, policy_key, False),
        ):
            if not length:
                continue
            trajectory, state = self._rollout(state, rollout_key, length, random)
            episodes = trajectory.dones.sum().astype(jnp.uint32)
            trajectory = tree_stop_gradient(
                flatten_rollout_trajectory(clean_trajectory(trajectory))
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
                agent_state=agent_state,
                replay_buffer_state=self.replay_buffer.add(
                    state.replay_buffer_state, trajectory
                ),
                metrics=state.metrics.replace(
                    sampled_timesteps=state.metrics.sampled_timesteps
                    + psum(
                        jnp.uint32(length * self.config.num_envs), self.dp_axis_name
                    ),
                    sampled_episodes=state.metrics.sampled_episodes
                    + psum(episodes, self.dp_axis_name),
                ),
            )
        if random_length + policy_length > 0:
            key, env_key = jax.random.split(state.key)
            if self.is_multi_device:
                env_key = jax.random.fold_in(
                    env_key, jax.lax.axis_index(self.dp_axis_name)
                )
            # Training starts a new episode, independent of replay prefill.
            state = state.replace(
                key=key,
                env_state=self.env.reset(env_key),
                rollout_preferences=training_preferences,
            )
        return state

    def evaluate(self, state):
        key, eval_key = jax.random.split(state.key)
        if self.is_multi_device:
            eval_key = jax.random.fold_in(
                eval_key, jax.lax.axis_index(self.dp_axis_name)
            )
        weights = jnp.asarray(self.eval_preferences, jnp.float32)

        def evaluate_preference(weight, key):
            agent_state = state.agent_state.replace(
                extra_state=state.agent_state.extra_state.replace(preferences=weight)
            )
            metrics = self.evaluator.evaluate(
                agent_state, key, self.config.eval_episodes
            )
            return metrics.episode_returns.mean(axis=0), metrics.episode_lengths.mean()

        returns, lengths = jax.vmap(evaluate_preference)(
            weights, jax.random.split(eval_key, len(self.eval_preferences))
        )
        metrics = MultiObjectiveEvaluateMetric(
            preferences=weights,
            episode_returns=returns,
            scalarized_returns=(weights * returns).sum(-1),
            episode_lengths=lengths,
        ).all_reduce(self.dp_axis_name)
        return metrics, state.replace(key=key)

    @classmethod
    def enable_shmap(cls):
        specs = State(
            key=jax.P(),
            metrics=jax.P(),
            agent_state=jax.P(),
            opt_state=jax.P(),
            env_state=jax.P(DP_AXIS_NAME),
            replay_buffer_state=jax.P(DP_AXIS_NAME),
            rollout_preferences=jax.P(DP_AXIS_NAME),
        )
        step_fn, warmup_fn = cls.step, cls._postsetup_replaybuffer

        def sharded_step(self, state):
            with _shard_map_replay_buffer_context(state) as local:
                metrics, next_state = step_fn(self, local)
                local.update(next_state)
            return metrics, local

        def sharded_warmup(self, state):
            with _shard_map_replay_buffer_context(state) as local:
                local.update(warmup_fn(self, local))
            return local

        cls.step = jax.shard_map(
            sharded_step, in_specs=(None, specs), out_specs=(jax.P(), specs)
        )
        cls._postsetup_replaybuffer = jax.shard_map(
            sharded_warmup, in_specs=(None, specs), out_specs=specs
        )
        cls.evaluate = jax.shard_map(
            cls.evaluate, in_specs=(None, specs), out_specs=(jax.P(), specs)
        )


def validate_replay_config(config):
    """Require enough real transitions before the first replay update."""
    if (
        min(
            config.num_envs,
            config.rollout_length,
            config.batch_size,
            config.num_updates_per_iter,
            config.fold_iters,
        )
        <= 0
    ):
        raise ValueError(
            "Environment, rollout, batch, update and fold counts must be positive."
        )
    if (
        min(
            config.random_timesteps,
            config.learning_start_timesteps,
            config.total_timesteps,
        )
        < 0
    ):
        raise ValueError("Sampling budgets must be nonnegative.")
    if config.replay_buffer_capacity < max(
        config.batch_size, config.rollout_length * config.num_envs
    ):
        raise ValueError("Replay capacity must fit a rollout and a training batch.")
    warmup = max(
        config.random_timesteps // config.num_envs * config.num_envs,
        config.learning_start_timesteps,
    )
    if warmup < config.batch_size:
        raise ValueError("Warmup must collect at least batch_size transitions.")
