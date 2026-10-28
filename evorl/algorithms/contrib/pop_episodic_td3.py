from evorl.utils import running_statistics
from flax import nnx
from functools import partial
import math
from typing import Self
from omegaconf import DictConfig

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.agent import Agent, AgentState
from evorl.types import PyTreeDict, State
from evorl.metrics import MetricBase, EvaluateMetric, WorkflowMetric
from evorl.replay_buffers import ReplayBuffer, AbstractReplayBuffer
from evorl.envs import create_env, AutoresetMode, BaseEnv
from evorl.utils.rl_toolkits import (
    soft_target_update,
)
from evorl.recorders import add_prefix, get_1d_array_statistics, get_1d_array
from evorl.evaluators import Evaluator, EpisodeCollector


from evorl.algorithms.offpolicy_utils import skip_replay_buffer_state
from evorl.algorithms.td3 import TD3Agent, TD3TrainMetric


import copy
from evorl.workflows import Workflow
from evorl.utils.jax_utils import scan_and_mean
from evorl.algorithms.episodic_utils import (
    init_replay_buffer,
    prefill_replay_buffer,
    collect_population_episodes,
)


class PopEpisodicTD3TrainMetric(MetricBase):
    """Independent TD3 population returns and training losses."""

    rb_size: jax.Array
    pop_episode_returns: jax.Array
    pop_episode_lengths: jax.Array
    rl_metrics: MetricBase


class PopEpisodicTD3Workflow(Workflow):
    """Independent TD3 learners sharing episodic collection and one replay."""

    def __init__(
        self,
        *,
        env: BaseEnv,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        collector: EpisodeCollector,
        evaluator: Evaluator,
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
        self.collector = collector
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

    def setup(self, key: jax.Array) -> State:
        key, agent_key, replay_key = jax.random.split(key, 3)
        agent_state, opt_state = self._setup_agent_and_optimizer(agent_key)
        state = State(
            key=key,
            agent_state=agent_state,
            opt_state=opt_state,
            replay_buffer_state=init_replay_buffer(
                self.env, self.replay_buffer, replay_key
            ),
            metrics=WorkflowMetric(),
        )
        return self._postsetup_replaybuffer(state)

    def _postsetup_replaybuffer(self, state: State) -> State:
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

    def _rollout(self, pop_agent_state, replay_buffer_state, key):
        return collect_population_episodes(
            pop_agent_state,
            replay_buffer_state,
            key,
            collector=self.collector,
            replay_buffer=self.replay_buffer,
            agent_state_vmap_axes=self.agent_state_vmap_axes,
            num_episodes=self.config.episodes_for_fitness,
            num_agents=self.config.pop_size,
        )

    def _rl_update(self, agent_state, opt_state, replay_buffer_state, key):
        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            key, rb_key, learn_key = jax.random.split(key, 3)

            rb_keys = jax.random.split(
                rb_key,
                self.config.actor_update_interval * self.config.pop_size,
            )
            sample_batches = jax.vmap(self.replay_buffer.sample, in_axes=(None, 0))(
                replay_buffer_state, rb_keys
            )

            # (actor_update_interval, pop_size, B, ...)
            sample_batches = jtu.tree_map(
                lambda x: x.reshape(
                    (
                        self.config.actor_update_interval,
                        self.config.pop_size,
                        *x.shape[1:],
                    )
                ),
                sample_batches,
            )

            (agent_state, opt_state), train_info = update_pop_td3(
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

        td3_metrics = td3_metrics.replace(
            actor_loss=td3_metrics.actor_loss / self.config.pop_size,
            critic_loss=td3_metrics.critic_loss / self.config.pop_size,
        )

        return td3_metrics, agent_state, opt_state

    @classmethod
    def enable_jit(cls) -> None:
        cls.step = jax.jit(cls.step, static_argnums=(0,))
        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )

    @classmethod
    def name(cls):
        return "PopEpisodicTD3"

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        assert config.random_timesteps > 0, (
            "random_timesteps should be positive to pre-fill some data in the replay buffer"
        )

        # env for one actor
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

        collector = EpisodeCollector(
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
            collector=collector,
            evaluator=evaluator,
            replay_buffer=replay_buffer,
            config=config,
        )

        return workflow

    def _setup_agent_and_optimizer(self, key: jax.Array):
        agent_state = jax.vmap(self.agent.init, in_axes=(None, None, 0))(
            self.env.obs_space,
            self.env.action_space,
            jax.random.split(key, self.config.pop_size),
        )

        if agent_state.obs_preprocessor_state is not None:
            # Every member starts with identical statistics; normalization is
            # shared across the population, matching agent_state_vmap_axes.
            agent_state = agent_state.replace(
                obs_preprocessor_state=jax.tree.map(
                    lambda x: x[0], agent_state.obs_preprocessor_state
                )
            )

        models = nnx.merge(agent_state.model_state, copy=True)
        opt_state = PyTreeDict(
            actor=nnx.unpack(
                nnx.vmap(
                    lambda model: nnx.Optimizer(model, self.optimizer, wrt=nnx.Param)
                )(models.actor)
            ),
            critic=nnx.unpack(
                nnx.vmap(
                    lambda model: nnx.Optimizer(model, self.optimizer, wrt=nnx.Param)
                )(models.critic)
            ),
        )

        return agent_state, opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        opt_state = state.opt_state
        replay_buffer_state = state.replay_buffer_state
        iterations = state.metrics.iterations + 1

        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        # ======== RL update ========
        td3_metrics, agent_state, opt_state = self._rl_update(
            agent_state,
            opt_state,
            replay_buffer_state,
            learn_key,
        )

        # the trajectory [T, #pop*B, ...]
        # metrics: [#pop, B]
        eval_metrics, trajectory, replay_buffer_state = self._rollout(
            agent_state, replay_buffer_state, rollout_key
        )

        train_metrics = PopEpisodicTD3TrainMetric(
            rb_size=replay_buffer_state.buffer_size,
            pop_episode_lengths=eval_metrics.episode_length.mean(-1),
            pop_episode_returns=eval_metrics.episode_return.mean(-1),
            rl_metrics=td3_metrics,
        )

        # calculate the number of timestep
        sampled_timesteps = eval_metrics.episode_length.sum().astype(jnp.uint32)
        sampled_episodes = jnp.uint32(self.config.episodes_for_fitness * pop_size)

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
            iterations=iterations,
        )

        state = state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
            opt_state=opt_state,
        )

        return train_metrics, state

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        key, eval_key = jax.random.split(state.key, num=2)

        # [#pop, #episodes]
        raw_eval_metrics = jax.vmap(
            partial(self.evaluator.evaluate, num_episodes=self.config.eval_episodes),
            in_axes=(self.agent_state_vmap_axes, 0),
        )(
            state.agent_state,
            jax.random.split(eval_key, self.config.pop_size),
        )

        eval_metrics = EvaluateMetric(
            episode_return=raw_eval_metrics.episode_return.mean(-1),
            episode_length=raw_eval_metrics.episode_length.mean(-1),
        )

        state = state.replace(key=key)
        return eval_metrics, state

    def learn(self, state: State) -> State:
        sampled_episodes = int(state.metrics.sampled_episodes)
        start_iteration = int(state.metrics.iterations)
        num_iters = math.ceil(
            (self.config.total_episodes - sampled_episodes)
            / (self.config.episodes_for_fitness * self.config.pop_size)
        )

        final_iteration = start_iteration + num_iters
        for i in range(start_iteration, final_iteration):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            workflow_metrics_dict = workflow_metrics.to_local_dict()
            self.recorder.write(workflow_metrics_dict, iters)

            train_metrics_dict = train_metrics.to_local_dict()
            train_metrics_dict["pop_episode_returns"] = get_1d_array_statistics(
                train_metrics_dict["pop_episode_returns"], histogram=True
            )

            train_metrics_dict["pop_episode_lengths"] = get_1d_array_statistics(
                train_metrics_dict["pop_episode_lengths"], histogram=True
            )

            if train_metrics_dict["rl_metrics"] is not None:
                train_metrics_dict["rl_metrics"]["raw_loss_dict"] = jtu.tree_map(
                    get_1d_array_statistics,
                    train_metrics_dict["rl_metrics"]["raw_loss_dict"],
                )

            self.recorder.write(train_metrics_dict, iters)

            if iters % self.config.eval_interval == 0 or iters == final_iteration:
                eval_metrics, state = self.evaluate(state)

                eval_metrics_dict = jtu.tree_map(
                    get_1d_array,
                    eval_metrics.to_local_dict(),
                )

                self.recorder.write(add_prefix(eval_metrics_dict, "eval"), iters)

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)

            self.checkpoint_manager.save(
                iters,
                saved_state,
                force=iters == final_iteration,
            )

        return state


def update_pop_td3(
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
    Batches have axes `(critic_updates, population, batch, ...)`; each member
    clips its gradients and updates its own actor/critic optimizer state.
    """
    models = agent.get_models(agent_state, copy=True)
    actor_optimizer = nnx.merge(opt_state.actor, copy=True)
    critic_optimizer = nnx.merge(opt_state.critic, copy=True)

    num_agents = config.pop_size
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
