"""Pareto Conditioned Networks (Reymond et al., AAMAS 2022).

PCN learns actions by supervised hindsight on complete episodes, conditioned
on return-to-go and remaining horizon. All objectives maximize. Deterministic
means that the full task state and action fix the next state and reward, without
fresh transition or reward randomness. Random resets and exploratory actions
do not violate this assumption.
Replaying the same RNG seed does not make a stochastic task deterministic.

EvoRL's toy DeepSeaTreasure satisfies it. Many Brax and MuJoCo Playground physics
tasks can also satisfy it with fixed physical parameters and perturbations
disabled; this is task-specific. FrozenLake with `is_slippery=True`, random
disturbances and random rewards are counterexamples. Selecting successful
trajectories in hindsight can select lucky outcomes in stochastic tasks:
imitating their actions does not guarantee their return in expectation.

The network, commands and replay priorities follow MORL-Baselines' PCN. A fixed
episode archive replaces its Python heap so training supports JIT and nested
vmap. Episodes are padded to `max_episode_steps`; only real transitions enter
the supervised loss and sampling counters. Replay capacity is in episodes.

Paper: https://arxiv.org/abs/2204.05036.
Reference: https://lucasalegre.github.io/morl-baselines/algos/multi_policy/pcn/.
"""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from evorl.agent import Agent, AgentState, ObsPreprocessorFn
from evorl.distributed import all_gather, pmean, pmean_nnx_loss_and_grad
from evorl.distributed.sharding import mark_varying
from evorl.envs import AutoresetMode, Box, Discrete, create_env
from evorl.metrics import MetricBase, TrainMetric
from evorl.recorders import add_prefix
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict, State, pytree_field
from evorl.utils import running_statistics
from evorl.utils.jax_utils import rng_fold_in_axis
from evorl.workflows.rl_workflow import RLWorkflow

from .common import MultiObjectiveWorkflowMetric, objective_count


def episode_returns(rewards, valid, discount):
    """Compute masked Monte Carlo returns for `(episodes, time, objectives)`."""

    def step(total, item):
        reward, active = item
        total = jnp.where(active[..., None], reward + discount * total, 0)
        return total, total

    _, returns = jax.lax.scan(
        step,
        mark_varying(
            jnp.zeros_like(rewards[:, 0]),
            jax.sharding.get_abstract_mesh().manual_axes,
        ),
        (jnp.swapaxes(rewards, 0, 1), valid.T),
        reverse=True,
    )
    return jnp.swapaxes(returns, 0, 1)


def episode_priorities(returns, valid, crowding_threshold=0.2):
    """Rank complete episodes by distance to the Pareto front and diversity.

    Invalid slots rank below every episode. Duplicate returns receive the
    reference's small penalty, retaining one representative. Dense points
    (normalized crowding below the threshold) receive twice their penalty.
    """
    dominates = (
        jnp.all(returns[:, None] >= returns[None, :], axis=-1)
        & jnp.any(returns[:, None] > returns[None, :], axis=-1)
        & valid[:, None]
    )
    front = valid & ~jnp.any(dominates, axis=0)
    distances = jnp.linalg.norm(returns[:, None] - returns[None, :], axis=-1)
    distance = jnp.min(jnp.where(front[None], distances, jnp.inf), axis=-1)
    equal = jnp.all(returns[:, None] == returns[None, :], axis=-1)
    earlier = jnp.arange(len(returns))[:, None] < jnp.arange(len(returns))[None, :]
    duplicate = jnp.any(equal & earlier & valid[:, None], axis=0)
    score = -distance - jnp.where(~front | duplicate, 1e-5, 0)

    low = jnp.where(
        valid.any(), jnp.min(jnp.where(valid[:, None], returns, jnp.inf), axis=0), 0
    )
    high = jnp.where(
        valid.any(), jnp.max(jnp.where(valid[:, None], returns, -jnp.inf), axis=0), 1
    )
    normalized = (returns - low) / (high - low + 1e-8)
    indices = jnp.argsort(jnp.where(valid[:, None], normalized, jnp.inf), axis=0)
    ordered = jnp.take_along_axis(normalized, indices, axis=0)
    count = valid.sum()
    ranks = jnp.arange(len(returns))[:, None]
    neighbors = jnp.abs(jnp.roll(ordered, -1, axis=0) - jnp.roll(ordered, 1, axis=0))
    gaps = jnp.where((ranks == 0) | (ranks >= count - 1), 1.0, neighbors)
    crowding = (
        jnp.zeros_like(returns)
        .at[indices, jnp.arange(returns.shape[-1])]
        .set(gaps)
        .sum(-1)
    )
    score = jnp.where(crowding <= crowding_threshold, 2 * score, score)
    return jnp.where(valid, score, -jnp.inf), front


def add_episodes(archive, episodes):
    """Insert each new complete episode, evicting the least useful old one.

    Fresh episodes receive positive priority, matching the reference heap
    insertion. Rank existing episodes once for the entire collection batch;
    a second fresh episode must not evict the first merely for being dominated.
    """
    priorities, _ = episode_priorities(archive.returns[:, 0], archive.lengths > 0)

    def add(carry, episode):
        archive, priorities = carry
        index = priorities.argmin()
        archive = jax.tree.map(lambda x, y: x.at[index].set(y), archive, episode)
        priorities = priorities.at[index].set(1)
        return (archive, priorities), None

    return jax.lax.scan(add, (archive, priorities), episodes)[0][0]


def sample_episode_batch(archive, key, batch_size):
    """Sample episodes uniformly, then a real transition uniformly per episode."""
    episode_key, time_key = jax.random.split(key)
    indices = jax.random.categorical(
        episode_key, jnp.where(archive.lengths > 0, 0.0, -jnp.inf), shape=(batch_size,)
    )
    lengths = archive.lengths[indices]
    times = jax.random.randint(time_key, (batch_size,), 0, lengths)
    return SampleBatch(
        obs=archive.obs[indices, times],
        actions=archive.actions[indices, times],
        extras=PyTreeDict(
            desired_return=archive.returns[indices, times],
            desired_horizon=(lengths - times).astype(jnp.float32),
        ),
    )


def sample_commands(archive, key, num_commands, num_candidates):
    """Pick a best nondominated episode and improve one return coordinate."""
    priorities, _ = episode_priorities(archive.returns[:, 0], archive.lengths > 0)
    _, indices = jax.lax.top_k(priorities, num_candidates)
    returns = archive.returns[indices, 0]
    _, front = episode_priorities(returns, archive.lengths[indices] > 0)
    choice_key, objective_key, noise_key = jax.random.split(key, 3)
    chosen = jax.random.categorical(
        choice_key, jnp.where(front, 0.0, -jnp.inf), shape=(num_commands,)
    )
    mean = jnp.sum(jnp.where(front[:, None], returns, 0), axis=0) / front.sum()
    variance = (
        jnp.sum(jnp.where(front[:, None], (returns - mean) ** 2, 0), axis=0)
        / front.sum()
    )
    objectives = jax.random.randint(
        objective_key, (num_commands,), 0, returns.shape[-1]
    )
    improve = (
        jax.random.uniform(noise_key, (num_commands,)) * jnp.sqrt(variance)[objectives]
    )
    desired = returns[chosen].at[jnp.arange(num_commands), objectives].add(improve)
    horizons = jnp.maximum(archive.lengths[indices[chosen]] - 2, 1).astype(jnp.float32)
    return desired, horizons


class PCNNetwork(nnx.Module):
    """Multiply state and command embeddings, following the released model."""

    def __init__(self, obs_size, action_size, objectives, hidden_size, *, rngs):
        self.state_embedding = nnx.Linear(obs_size, hidden_size, rngs=rngs)
        self.command_embedding = nnx.Linear(objectives + 1, hidden_size, rngs=rngs)
        self.hidden = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.output = nnx.Linear(hidden_size, action_size, rngs=rngs)

    def __call__(self, obs, commands):
        hidden = jax.nn.sigmoid(self.state_embedding(obs)) * jax.nn.sigmoid(
            self.command_embedding(commands)
        )
        return self.output(jax.nn.relu(self.hidden(hidden)))


class PCNAgent(Agent):
    """An action policy conditioned on desired return and remaining horizon."""

    num_objectives: int = pytree_field(static=True)
    agent_network_cfg: object = pytree_field(static=True)
    scaling_factor: tuple = pytree_field(static=True)
    obs_preprocessor: ObsPreprocessorFn | None = pytree_field(default=None, static=True)
    noise: float = 0.1

    def init(self, obs_space, action_space, key):
        continuous = isinstance(action_space, Box)
        if not continuous and not isinstance(action_space, Discrete):
            raise TypeError(
                "PCN supports scalar Discrete or one-dimensional Box actions."
            )
        if continuous and len(action_space.shape) != 1:
            raise TypeError("PCN continuous actions must be one-dimensional.")
        action_size = action_space.shape[0] if continuous else action_space.n
        policy = PCNNetwork(
            obs_space.shape[0],
            action_size,
            self.num_objectives,
            self.agent_network_cfg.hidden_size,
            rngs=nnx.Rngs(key),
        )
        return AgentState(
            model_state=nnx.unpack(nnx.Dict(policy=policy)),
            obs_preprocessor_state=running_statistics.init_state(
                jnp.zeros(obs_space.shape, jnp.float32)
            )
            if self.obs_preprocessor is not None
            else None,
            extra_state=PyTreeDict(
                action_low=jnp.asarray(action_space.low, jnp.float32)
                if continuous
                else None,
                action_high=jnp.asarray(action_space.high, jnp.float32)
                if continuous
                else None,
                desired_return=jnp.zeros(self.num_objectives, jnp.float32),
                desired_horizon=jnp.float32(1),
            ),
        )

    def predictions(self, agent_state, sample_batch):
        """Predict logits or continuous actions from explicit commands."""
        obs = sample_batch.obs.astype(jnp.float32)
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        extras = sample_batch.extras or agent_state.extra_state
        returns = jnp.broadcast_to(
            extras.desired_return, (*obs.shape[:-1], self.num_objectives)
        )
        horizons = jnp.broadcast_to(extras.desired_horizon, obs.shape[:-1])
        command = jnp.concatenate((returns, horizons[..., None]), axis=-1)
        command = command * jnp.asarray(self.scaling_factor, jnp.float32)
        return self.get_models(agent_state).policy(obs, command)

    def compute_actions(self, agent_state, sample_batch, key):
        prediction = self.predictions(agent_state, sample_batch)
        if agent_state.extra_state.action_low is not None:
            action = jnp.clip(
                prediction + self.noise * jax.random.normal(key, prediction.shape),
                agent_state.extra_state.action_low,
                agent_state.extra_state.action_high,
            )
        else:
            action = jax.random.categorical(key, prediction)
        return action, PyTreeDict()

    def evaluate_actions(self, agent_state, sample_batch, key):
        prediction = self.predictions(agent_state, sample_batch)
        action = (
            jnp.clip(
                prediction,
                agent_state.extra_state.action_low,
                agent_state.extra_state.action_high,
            )
            if agent_state.extra_state.action_low is not None
            else prediction.argmax(-1)
        )
        return action, PyTreeDict()

    def loss(self, agent_state, sample_batch, key):
        """Reproduce hindsight actions with cross-entropy or continuous MSE."""
        prediction = self.predictions(agent_state, sample_batch)
        if agent_state.extra_state.action_low is not None:
            loss = jnp.square(prediction - sample_batch.actions).mean()
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                prediction, sample_batch.actions
            ).mean()
        return PyTreeDict(policy_loss=loss)


class PCNEvaluateMetric(MetricBase):
    """Evaluate each requested return/horizon command without scalarization.

    `desired_returns` and `episode_return` have shape `(commands, objectives)`;
    `desired_horizons` and `episode_length` have shape `(commands,)`. Return
    and length keep the common per-policy metric names; the leading command
    axis is retained.
    """

    desired_returns: jax.Array
    desired_horizons: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    command_error: jax.Array


class PCNWorkflow(RLWorkflow):
    """Train PCN on complete episodes with a functional trajectory archive.

    Both the archive and NNX variables live in `State`. Training and evaluation
    support nested vmap and JIT, including native and host-callback environments.
    Data-parallel training keeps one global episode archive replicated across
    devices. Complete episodes are gathered before common retention and command
    selection; each device samples its training batch from the whole archive.
    Collection finishes whole episode batches, so the sampling budget can
    overshoot by one such batch.
    """

    def __init__(self, env, eval_env, agent, optimizer, config):
        super().__init__(config)
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        self.optimizer = optimizer

    @classmethod
    def name(cls):
        return "PCN"

    @classmethod
    def _rescale_config(cls, config):
        devices = jax.local_device_count()
        if config.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if config.num_initial_episodes % config.num_envs:
            raise ValueError("num_initial_episodes must be divisible by num_envs.")
        if config.episode_buffer_capacity < config.num_envs:
            raise ValueError(
                "episode_buffer_capacity must fit a complete collection batch."
            )
        for field in ("num_envs", "num_eval_envs", "batch_size"):
            if config[field] < devices or config[field] % devices:
                raise ValueError(
                    f"{field} must be positive and divisible by num_devices."
                )
            config[field] //= devices

    @classmethod
    def _build_from_config(cls, config):
        for field in (
            "num_envs",
            "num_eval_envs",
            "num_initial_episodes",
            "episode_buffer_capacity",
            "batch_size",
            "num_updates_per_iter",
            "num_command_candidates",
            "num_eval_commands",
            "eval_interval",
        ):
            if config[field] <= 0:
                raise ValueError(f"{field} must be positive.")
        if config.num_initial_episodes % config.num_envs:
            raise ValueError("num_initial_episodes must be divisible by num_envs.")
        if config.episode_buffer_capacity < config.num_envs:
            raise ValueError(
                "episode_buffer_capacity must fit a complete collection batch."
            )
        if max(config.num_command_candidates, config.num_eval_commands) > min(
            config.episode_buffer_capacity, config.num_initial_episodes
        ):
            raise ValueError(
                "Command counts must fit the initially populated episode archive."
            )
        if not 0 < config.discount <= 1 or config.total_timesteps < 0:
            raise ValueError(
                "discount must be in (0, 1] and total_timesteps nonnegative."
            )
        env = create_env(
            config.env, parallel=config.num_envs, autoreset_mode=AutoresetMode.DISABLED
        )
        eval_env = create_env(
            config.env,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )
        objectives = objective_count(env)
        scaling = config.scaling_factor
        if scaling is None:
            scaling = [1.0] * objectives + [1 / config.env.max_episode_steps]
        if (
            len(scaling) != objectives + 1
            or not np.all(np.isfinite(scaling))
            or np.any(np.asarray(scaling) <= 0)
        ):
            raise ValueError(
                "scaling_factor needs a positive finite scale per objective and horizon."
            )
        if config.max_return is not None and (
            len(config.max_return) != objectives
            or not np.all(np.isfinite(config.max_return))
        ):
            raise ValueError("max_return needs a finite upper bound per objective.")
        if isinstance(env.action_space, Box) and not (
            np.isfinite(env.action_space.low).all()
            and np.isfinite(env.action_space.high).all()
        ):
            raise ValueError("PCN requires finite continuous action bounds.")
        agent = PCNAgent(
            num_objectives=objectives,
            agent_network_cfg=config.agent_network,
            scaling_factor=tuple(scaling),
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
            noise=config.exploration_noise,
        )
        return cls(env, eval_env, agent, optax.adam(config.optimizer.lr), config)

    def _collect_episodes(
        self,
        env,
        agent_state,
        key,
        desired_return,
        desired_horizon,
        *,
        random=False,
        evaluate=False,
    ):
        key = rng_fold_in_axis(key, self.dp_axis_name)
        key, reset_key = jax.random.split(key)
        env_state = env.reset(reset_key)
        desired_return = jnp.broadcast_to(
            desired_return, (env.num_envs, self.agent.num_objectives)
        )
        desired_horizon = jnp.broadcast_to(desired_horizon, (env.num_envs,))
        upper = (
            jnp.asarray(self.config.max_return, jnp.float32)
            if self.config.max_return is not None
            else jnp.full((self.agent.num_objectives,), jnp.inf)
        )

        def step(carry, _):
            state, key, desired_return, horizon = carry
            key, action_key = jax.random.split(key)
            if random:
                actions = jax.vmap(env.action_space.sample)(
                    jax.random.split(action_key, env.num_envs)
                )
            else:
                batch = SampleBatch(
                    obs=state.obs,
                    extras=PyTreeDict(
                        desired_return=desired_return, desired_horizon=horizon
                    ),
                )
                action_fn = (
                    self.agent.evaluate_actions
                    if evaluate
                    else self.agent.compute_actions
                )
                actions, _ = action_fn(agent_state, batch, action_key)
            next_state = env.step(state, actions)
            active = ~state.done.astype(jnp.bool_)
            rewards = jnp.where(active[..., None], next_state.reward, 0)
            record = PyTreeDict(
                obs=state.obs,
                next_obs=next_state.obs,
                actions=actions,
                rewards=rewards,
                valid=active,
            )
            desired_return = jnp.where(
                active[..., None],
                jnp.minimum((desired_return - rewards) / self.config.discount, upper),
                desired_return,
            )
            horizon = jnp.where(active, jnp.maximum(horizon - 1, 1), horizon)
            return (next_state, key, desired_return, horizon), record

        _, trajectory = jax.lax.scan(
            step,
            mark_varying(
                (env_state, key, desired_return, desired_horizon),
                jax.sharding.get_abstract_mesh().manual_axes,
            ),
            (),
            length=self.config.env.max_episode_steps,
        )
        episodes = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), trajectory)
        episodes.lengths = episodes.valid.sum(-1).astype(jnp.int32)
        episodes.returns = episode_returns(
            episodes.rewards, episodes.valid, self.config.discount
        )
        return episodes

    def setup(self, key):
        key, agent_key, buffer_key = jax.random.split(key, 3)
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )
        models = self.agent.get_models(agent_state, copy=True)
        opt_state = nnx.unpack(
            nnx.Optimizer(models.policy, self.optimizer, wrt=nnx.Param)
        )
        zero_return = jnp.zeros(self.agent.num_objectives, jnp.float32)
        template = jax.eval_shape(
            lambda key: self._collect_episodes(
                self.env, agent_state, key, zero_return, jnp.float32(1), random=True
            ),
            buffer_key,
        )
        archive = jax.tree.map(
            lambda x: jnp.zeros(
                (self.config.episode_buffer_capacity, *x.shape[1:]), x.dtype
            ),
            template,
        )

        def warmup(carry, _):
            archive, agent_state, key, timesteps = carry
            key, rollout_key = jax.random.split(key)
            episodes = self._collect_episodes(
                self.env,
                agent_state,
                rollout_key,
                zero_return,
                jnp.float32(1),
                random=True,
            )
            if agent_state.obs_preprocessor_state is not None:
                agent_state = agent_state.replace(
                    obs_preprocessor_state=running_statistics.update(
                        agent_state.obs_preprocessor_state,
                        episodes.obs.astype(jnp.float32),
                        weights=episodes.valid,
                        axis_name=self.dp_axis_name,
                    )
                )
            if self.is_multi_device:
                episodes = jax.tree.map(
                    lambda x: all_gather(
                        x, self.dp_axis_name, axis=0, tiled=True, to="invarying"
                    ),
                    episodes,
                )
            return (
                add_episodes(archive, episodes),
                agent_state,
                key,
                timesteps + episodes.lengths.sum().astype(jnp.uint32),
            ), None

        (archive, agent_state, key, timesteps), _ = jax.lax.scan(
            warmup,
            (archive, agent_state, key, jnp.uint32(0)),
            (),
            length=self.config.num_initial_episodes
            // (self.config.num_envs * self.num_devices),
        )
        return State(
            key=key,
            agent_state=agent_state,
            opt_state=opt_state,
            episode_archive=archive,
            metrics=MultiObjectiveWorkflowMetric(
                sampled_timesteps=timesteps,
            ),
        )

    def step(self, state):
        def update(carry, _):
            agent_state, opt_state, key = carry
            key, sample_key = jax.random.split(key)
            sample_key = rng_fold_in_axis(sample_key, self.dp_axis_name)
            batch = sample_episode_batch(
                state.episode_archive, sample_key, self.config.batch_size
            )
            models = self.agent.get_models(agent_state, copy=True)

            def loss_fn(models):
                return self.agent.loss(
                    agent_state.replace(model_state=nnx.unpack(models)),
                    batch,
                    sample_key,
                ).policy_loss

            loss, grads = pmean_nnx_loss_and_grad(loss_fn, self.dp_axis_name)(models)
            optimizer = nnx.merge(opt_state, copy=True)
            optimizer.update(models.policy, grads.policy)
            return (
                agent_state.replace(model_state=nnx.unpack(models)),
                nnx.unpack(optimizer),
                key,
            ), loss

        (agent_state, opt_state, key), losses = jax.lax.scan(
            update,
            (state.agent_state, state.opt_state, state.key),
            (),
            length=self.config.num_updates_per_iter,
        )
        key, command_key, rollout_key = jax.random.split(key, 3)
        command_key = rng_fold_in_axis(command_key, self.dp_axis_name)
        desired, horizons = sample_commands(
            state.episode_archive,
            command_key,
            self.config.num_envs,
            self.config.num_command_candidates,
        )
        episodes = self._collect_episodes(
            self.env, agent_state, rollout_key, desired, horizons
        )
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    episodes.obs.astype(jnp.float32),
                    weights=episodes.valid,
                    axis_name=self.dp_axis_name,
                )
            )
        if self.is_multi_device:
            episodes = jax.tree.map(
                lambda x: all_gather(
                    x, self.dp_axis_name, axis=0, tiled=True, to="invarying"
                ),
                episodes,
            )
        metrics = state.metrics.replace(
            iterations=state.metrics.iterations + 1,
            training_updates=state.metrics.training_updates
            + self.config.num_updates_per_iter,
            sampled_timesteps=state.metrics.sampled_timesteps
            + episodes.lengths.sum().astype(jnp.uint32),
            sampled_episodes=state.metrics.sampled_episodes
            + self.config.num_envs * self.num_devices,
        )
        state = state.replace(
            key=key,
            agent_state=agent_state,
            opt_state=opt_state,
            episode_archive=add_episodes(state.episode_archive, episodes),
            metrics=metrics,
        )
        return TrainMetric(
            loss=losses.mean(),
            train_episode_return=episodes.returns[:, 0].mean(0),
            raw_loss_dict=PyTreeDict(policy_loss=losses.mean()),
        ), state

    def evaluate(self, state):
        key, eval_key = jax.random.split(state.key)
        priorities, _ = episode_priorities(
            state.episode_archive.returns[:, 0], state.episode_archive.lengths > 0
        )
        _, indices = jax.lax.top_k(priorities, self.config.num_eval_commands)
        desired = state.episode_archive.returns[indices, 0]
        horizons = state.episode_archive.lengths[indices].astype(jnp.float32)

        def evaluate_command(desired, horizon, key):
            episodes = self._collect_episodes(
                self.eval_env, state.agent_state, key, desired, horizon, evaluate=True
            )
            return episodes.returns[:, 0].mean(0), episodes.lengths.mean()

        returns, lengths = jax.vmap(evaluate_command)(
            desired, horizons, jax.random.split(eval_key, len(indices))
        )
        returns, lengths = pmean((returns, lengths), self.dp_axis_name)
        return PCNEvaluateMetric(
            desired_returns=desired,
            desired_horizons=horizons,
            episode_return=returns,
            episode_length=lengths,
            command_error=jnp.linalg.norm(returns - desired, axis=-1),
        ), state.replace(key=key)

    @classmethod
    def enable_shmap(cls):
        # All dynamic state is replicated: only collected episodes and training
        # samples are local. Explicit collectives reconcile the global archive
        # and model before crossing these boundaries.
        cls.setup = jax.shard_map(
            cls.setup, in_specs=(None, jax.P()), out_specs=jax.P()
        )
        cls.step = jax.shard_map(
            cls.step, in_specs=(None, jax.P()), out_specs=(jax.P(), jax.P())
        )
        cls.evaluate = jax.shard_map(
            cls.evaluate, in_specs=(None, jax.P()), out_specs=(jax.P(), jax.P())
        )

    def learn(self, state):
        while int(state.metrics.sampled_timesteps) < self.config.total_timesteps:
            train_metrics, state = self.step(state)
            iteration = int(state.metrics.iterations)
            final = int(state.metrics.sampled_timesteps) >= self.config.total_timesteps
            self.recorder.write(
                add_prefix(train_metrics.to_local_dict(), "train"), iteration
            )
            self.recorder.write(
                add_prefix(state.metrics.to_local_dict(), "workflow"), iteration
            )
            if iteration % self.config.eval_interval == 0 or final:
                eval_metrics, state = self.evaluate(state)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iteration
                )
            self.checkpoint_manager.save(iteration, state, force=final)
        return state
