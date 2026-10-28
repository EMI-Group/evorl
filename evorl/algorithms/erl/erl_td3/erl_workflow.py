import copy
import logging
from typing import Self
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax import nnx

from evorl.metrics import MetricBase, metric_field
from evorl.types import PyTreeDict, State
from evorl.evaluators import Evaluator, EpisodeCollector
from evorl.replay_buffers import AbstractReplayBuffer, ReplayBuffer
from evorl.agent import Agent, AgentState
from evorl.envs import create_env, AutoresetMode, BaseEnv
from evorl.workflows import Workflow
from evorl.ec.optimizers import EvoOptimizer, ECState
from evorl.algorithms.td3 import TD3Agent, TD3TrainMetric
from evorl.utils import running_statistics
from evorl.utils.jax_utils import scan_and_mean
from evorl.utils.rl_toolkits import soft_target_update
from evorl.algorithms.episodic_utils import (
    init_replay_buffer,
    prefill_replay_buffer,
    collect_population_episodes,
    import_td3_actor_params,
)

logger = logging.getLogger(__name__)


class ERLTrainMetric(MetricBase):
    pop_episode_return: jax.Array | None = None
    pop_episode_length: jax.Array | None = None
    rb_size: jax.Array | None = None
    rl_episode_return: jax.Array | None = None
    rl_episode_length: jax.Array | None = None
    rl_metrics: MetricBase | None = None
    ec_info: PyTreeDict = metric_field(default_factory=PyTreeDict)


class WorkflowMetric(MetricBase):
    sampled_timesteps: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )
    sampled_episodes: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )
    rl_sampled_timesteps: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )
    iterations: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )


class ERLWorkflowBase(Workflow):
    """Shared lifecycle for ERL populations with independent TD3 learners."""

    def __init__(
        self,
        *,
        env: BaseEnv,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        ec_optimizer: EvoOptimizer,
        ec_collector: EpisodeCollector,
        rl_collector: EpisodeCollector,
        evaluator: Evaluator,  # to evaluate the pop-mean actor
        replay_buffer: AbstractReplayBuffer,
        config: DictConfig,
    ):
        super().__init__(config)
        self.env = env
        self.agent = agent
        self.agent_state_vmap_axes = AgentState(
            model_state=0, obs_preprocessor_state=None
        )
        self.optimizer = optimizer
        self.ec_optimizer = ec_optimizer
        self.ec_collector = ec_collector
        self.rl_collector = rl_collector
        self.evaluator = evaluator
        self.replay_buffer = replay_buffer

    @classmethod
    def build_from_config(
        cls,
        config: DictConfig,
        multi_device: bool = False,
        jit: bool = True,
    ) -> Self:
        config = copy.deepcopy(config)  # avoid in-place modification

        devices = jax.local_devices()

        if multi_device or len(devices) > 1:
            raise NotImplementedError("Multi-devices is not supported yet.")

        if jit:
            cls.enable_jit()

        workflow = cls._build_from_config(config)

        return workflow

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        ec_optimizer = cls._build_ec_optimizer(config)

        # env for rl&ec rollout
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
            record_ori_obs=True,
        )

        agent = TD3Agent(
            agent_network_cfg=config.agent_network,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            critics_in_actor_loss=config.critics_in_actor_loss,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
        )

        if (
            config.optimizer.grad_clip_norm is not None
            and config.optimizer.grad_clip_norm > 0
        ):
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.optimizer.grad_clip_norm),
                optax.adam(config.optimizer.lr),
            )
        else:
            optimizer = optax.adam(config.optimizer.lr)

        if config.fitness_with_exploration:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        ec_collector = EpisodeCollector(
            env=env,
            action_fn=action_fn,
            max_episode_steps=config.env.max_episode_steps,
            env_extra_fields=("ori_obs", "termination"),
        )

        if config.rl_exploration:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        rl_collector = EpisodeCollector(
            env=env,
            action_fn=action_fn,
            max_episode_steps=config.env.max_episode_steps,
            env_extra_fields=("ori_obs", "termination"),
        )

        replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            min_sample_timesteps=config.batch_size,
            sample_batch_size=config.batch_size,
        )

        # to evaluate the pop-mean actor
        eval_env = create_env(
            config.env,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        workflow = cls(
            env=env,
            agent=agent,
            optimizer=optimizer,
            ec_optimizer=ec_optimizer,
            ec_collector=ec_collector,
            rl_collector=rl_collector,
            evaluator=evaluator,
            replay_buffer=replay_buffer,
            config=config,
        )

        return workflow

    @classmethod
    def _build_ec_optimizer(cls, config: DictConfig) -> EvoOptimizer:
        """Create the evolutionary optimizer selected by this variant."""
        raise NotImplementedError

    def setup(self, key: jax.Array) -> State:
        key, agent_key, rb_key = jax.random.split(key, 3)

        agent_state, opt_state, ec_opt_state = self._setup_agent_and_optimizer(
            agent_key
        )
        workflow_metrics = self._setup_workflow_metrics()
        replay_buffer_state = init_replay_buffer(self.env, self.replay_buffer, rb_key)

        state = State(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            opt_state=opt_state,
            ec_opt_state=ec_opt_state,
            replay_buffer_state=replay_buffer_state,
        )

        if self.config.random_timesteps > 0:
            logger.info("Start replay buffer post-setup")
            state = self._postsetup_replaybuffer(state)
            logger.info("Complete replay buffer post-setup")

        if self.config.warmup_iters > 0:
            logger.info("Start warmup")

            def _warmup_step(state, unused_t):
                _, state = self.warmup_step(state)
                return state, None

            state, _ = jax.lax.scan(
                _warmup_step, state, None, length=self.config.warmup_iters
            )

            logger.info("Complete warmup")

        return state

    def _setup_agent_and_optimizer(
        self, key: jax.Array
    ) -> tuple[AgentState, chex.ArrayTree, ECState]:
        raise NotImplementedError

    def _setup_workflow_metrics(self) -> MetricBase:
        return WorkflowMetric()

    def _postsetup_replaybuffer(self, state):
        env = create_env(
            self.config.env,
            parallel=self.config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )
        agent_state, buffer_state, key = prefill_replay_buffer(
            env,
            self.replay_buffer,
            state.agent_state,
            state.replay_buffer_state,
            state.key,
            self.config.random_timesteps,
        )
        return state.replace(
            key=key, agent_state=agent_state, replay_buffer_state=buffer_state
        )

    def warmup_step(self, state: State) -> tuple[ERLTrainMetric, State]:
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        ec_opt_state = state.ec_opt_state
        replay_buffer_state = state.replay_buffer_state

        key, ec_rollout_key = jax.random.split(state.key, 2)

        # 1. ask()
        pop_actor_params, ec_opt_state = self.ec_optimizer.ask(ec_opt_state)
        # 2. evaluate()
        pop_agent_state = make_erl_rollout_state(agent_state, pop_actor_params)
        ec_eval_metrics, ec_trajectory, replay_buffer_state = self._ec_rollout(
            pop_agent_state, replay_buffer_state, ec_rollout_key
        )
        fitnesses = ec_eval_metrics.episode_return.mean(axis=-1)
        # 3. tell()
        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        train_metrics = ERLTrainMetric(
            pop_episode_length=ec_eval_metrics.episode_length.mean(-1),
            pop_episode_return=ec_eval_metrics.episode_return.mean(-1),
            ec_info=ec_metrics,
            rb_size=replay_buffer_state.buffer_size,
        )

        # calculate the number of timestep
        sampled_timesteps = ec_eval_metrics.episode_length.sum().astype(jnp.uint32)
        sampled_episodes = jnp.uint32(self.config.episodes_for_fitness * pop_size)

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
            iterations=state.metrics.iterations + 1,
        )

        state = state.replace(
            key=key,
            metrics=workflow_metrics,
            replay_buffer_state=replay_buffer_state,
            ec_opt_state=ec_opt_state,
        )

        return train_metrics, state

    def _ec_rollout(self, agent_state, replay_buffer_state, key):
        return collect_population_episodes(
            agent_state,
            replay_buffer_state,
            key,
            collector=self.ec_collector,
            replay_buffer=self.replay_buffer,
            agent_state_vmap_axes=self.agent_state_vmap_axes,
            num_agents=self.config.pop_size,
            num_episodes=self.config.episodes_for_fitness,
        )

    def _rl_rollout(self, agent_state, replay_buffer_state, key):
        return collect_population_episodes(
            agent_state,
            replay_buffer_state,
            key,
            collector=self.rl_collector,
            replay_buffer=self.replay_buffer,
            agent_state_vmap_axes=self.agent_state_vmap_axes,
            num_agents=self.config.num_rl_agents,
            num_episodes=self.config.rollout_episodes,
        )

    def _rl_update(self, agent_state, opt_state, replay_buffer_state, key):
        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            key, rb_key, learn_key = jax.random.split(key, 3)

            rb_keys = jax.random.split(
                rb_key, self.config.actor_update_interval * self.config.num_rl_agents
            )
            sample_batches = jax.vmap(self.replay_buffer.sample, in_axes=(None, 0))(
                replay_buffer_state, rb_keys
            )

            # (actor_update_interval, num_rl_agents, B, ...)
            sample_batches = jtu.tree_map(
                lambda x: x.reshape(
                    (
                        self.config.actor_update_interval,
                        self.config.num_rl_agents,
                        *x.shape[1:],
                    )
                ),
                sample_batches,
            )

            (agent_state, opt_state), train_info = update_erl_td3(
                agent_state,
                opt_state,
                sample_batches,
                learn_key,
                agent=self.agent,
                agent_state_vmap_axes=self.agent_state_vmap_axes,
                config=self.config,
            )

            return (key, agent_state, opt_state), train_info

        (
            (_, agent_state, opt_state),
            (
                critic_loss,
                actor_loss,
                critic_loss_dict,
                actor_loss_dict,
            ),
        ) = scan_and_mean(
            _sample_and_update_fn,
            (key, agent_state, opt_state),
            (),
            length=self.config.num_rl_updates_per_iter,
        )

        # smoothed td3 metrics
        td3_metrics = TD3TrainMetric(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
        )

        return td3_metrics, agent_state, opt_state

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        raise NotImplementedError

    @classmethod
    def enable_jit(cls) -> None:
        cls.step = jax.jit(cls.step, static_argnums=(0,))

        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )
        cls.warmup_step = jax.jit(cls.warmup_step, static_argnums=(0,))


def make_erl_rollout_state(agent_state: AgentState, actor_params) -> AgentState:
    """Build actor-only population state without changing the RL learners.

    Evolutionary candidates need neither critics nor target actors. Import only
    online actor parameters and broadcast their auxiliary baseline for rollout.
    """
    models = nnx.merge(agent_state.model_state, copy=True)
    import_td3_actor_params(models.actor, actor_params)
    return agent_state.replace(model_state=nnx.unpack(nnx.Dict(actor=models.actor)))


def update_erl_td3(
    agent_state: AgentState,
    opt_state: PyTreeDict,
    sample_batches,
    key: jax.Array,
    *,
    agent: Agent,
    agent_state_vmap_axes: AgentState,
    config: DictConfig,
):
    """Update independent TD3 learners with member-local optimizers.

    Restore isolated NNX objects once, update them in place, then return snapshots.
    Forward-pass auxiliary variables are retained for the network being trained.
    Batches have axes `(critic_updates, learners, batch, ...)`. Each learner
    clips its own gradients and keeps independent actor/critic optimizer state.
    """
    models = agent.get_models(agent_state, copy=True)
    actor_optimizer = nnx.merge(opt_state.actor, copy=True)
    critic_optimizer = nnx.merge(opt_state.critic, copy=True)

    num_agents = config.num_rl_agents
    model_axes = agent_state_vmap_axes.model_state

    def critic_loss_fn(models, sample_batch, key):
        def member_loss(member, obs_state, batch, member_key):
            return agent.critic_loss(
                agent_state.replace(
                    model_state=nnx.unpack(member), obs_preprocessor_state=obs_state
                ),
                batch,
                member_key,
            )

        losses = nnx.vmap(
            member_loss,
            in_axes=(model_axes, agent_state_vmap_axes.obs_preprocessor_state, 0, 0),
        )(
            models,
            agent_state.obs_preprocessor_state,
            sample_batch,
            jax.random.split(key, num_agents),
        )
        return losses.critic_loss.sum(), losses

    def actor_loss_fn(models, sample_batch, key):
        def member_loss(member, obs_state, batch, member_key):
            return agent.actor_loss(
                agent_state.replace(
                    model_state=nnx.unpack(member), obs_preprocessor_state=obs_state
                ),
                batch,
                member_key,
            )

        losses = nnx.vmap(
            member_loss,
            in_axes=(model_axes, agent_state_vmap_axes.obs_preprocessor_state, 0, 0),
        )(
            models,
            agent_state.obs_preprocessor_state,
            sample_batch,
            jax.random.split(key, num_agents),
        )
        return losses.actor_loss.sum(), losses

    def update_critic(models, optimizer, sample_batch, key):
        value, grads = nnx.value_and_grad(
            critic_loss_fn,
            argnums=nnx.DiffState(0, nnx.All(nnx.Param, nnx.PathContains("critic"))),
            has_aux=True,
        )(models, sample_batch, key)
        nnx.vmap(lambda opt, model, grad: opt.update(model, grad))(
            optimizer, models.critic, grads["critic"]
        )
        return value

    key, critic_key, actor_key = jax.random.split(key, 3)
    if config.actor_update_interval > 1:

        def update_critics(carry, sample_batch):
            key, models, optimizer = carry
            key, update_key = jax.random.split(key)
            update_critic(models, optimizer, sample_batch, update_key)
            return key, models, optimizer

        _, multiple_update_key = jax.random.split(key)
        _, models, critic_optimizer = nnx.scan(
            update_critics, in_axes=(nnx.Carry, 0), out_axes=nnx.Carry
        )(
            (multiple_update_key, models, critic_optimizer),
            jax.tree.map(lambda x: x[:-1], sample_batches),
        )

    last_batch = jax.tree.map(lambda x: x[-1], sample_batches)
    critic_loss, critic_losses = update_critic(
        models, critic_optimizer, last_batch, critic_key
    )

    # Actor gradients use the critic without retaining changes to its auxiliary state.
    critic_auxiliary = nnx.clone(nnx.state(models.critic, nnx.Not(nnx.Param)))
    (actor_loss, actor_losses), grads = nnx.value_and_grad(
        actor_loss_fn,
        argnums=nnx.DiffState(0, nnx.All(nnx.Param, nnx.PathContains("actor"))),
        has_aux=True,
    )(models, last_batch, actor_key)
    nnx.vmap(lambda opt, model, grad: opt.update(model, grad))(
        actor_optimizer, models.actor, grads["actor"]
    )
    nnx.update(models.critic, critic_auxiliary)

    for target, online in (
        (models.target_actor, models.actor),
        (models.target_critic, models.critic),
    ):
        nnx.update(
            target,
            soft_target_update(
                nnx.state(target, nnx.Param), nnx.state(online, nnx.Param), config.tau
            ),
        )

    agent_state = agent_state.replace(model_state=nnx.unpack(models))
    opt_state = opt_state.replace(
        actor=nnx.unpack(actor_optimizer), critic=nnx.unpack(critic_optimizer)
    )
    return (
        (agent_state, opt_state),
        (critic_loss, actor_loss, critic_losses, actor_losses),
    )
