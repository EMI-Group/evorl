from flax import nnx
import logging
import math
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import optax

from evorl.metrics import MetricBase
from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import (
    tree_stop_gradient,
    right_shift_with_padding,
    scan_and_mean,
)
from evorl.utils.rl_toolkits import soft_target_update, flatten_rollout_trajectory
from evorl.agent import AgentState, Agent
from evorl.recorders import get_1d_array_statistics, add_prefix
from evorl.ec.optimizers import ECState, OpenES, ExponentialScheduleSpec
from evorl.algorithms.td3 import TD3TrainMetric
from evorl.algorithms.offpolicy_utils import clean_trajectory, skip_replay_buffer_state

from evorl.algorithms.erl.erl_td3.erl_workflow import (
    ERLTrainMetric,
    make_erl_rollout_state,
    ERLWorkflowBase,
)

logger = logging.getLogger(__name__)


class EvaluateMetric(MetricBase):
    rl_episode_returns: jax.Array
    rl_episode_lengths: jax.Array
    pop_center_episode_returns: jax.Array
    pop_center_episode_lengths: jax.Array


class ERLEDAWorkflow(ERLWorkflowBase):
    """ERL w/ EDA.

    Configs:

    - EC: n actors
    - RL: 1 (actor,critic)
    - Shared replay buffer

    RL will be injected into the pop mean. Support all EDA based ES algorithms.
    """

    @classmethod
    def name(cls):
        return "ERL-EDA"

    @classmethod
    def _build_ec_optimizer(cls, config: DictConfig):
        return OpenES(
            pop_size=config.pop_size,
            lr_schedule=ExponentialScheduleSpec(**config.ec_lr),
            noise_std_schedule=ExponentialScheduleSpec(**config.ec_noise_std),
            mirror_sampling=config.mirror_sampling,
        )

    def _setup_agent_and_optimizer(
        self, key: jax.Array
    ) -> tuple[AgentState, chex.ArrayTree, ECState]:
        agent_key, ec_key = jax.random.split(key)

        # one agent for RL
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )

        init_actor_params = nnx.state(
            nnx.merge(agent_state.model_state, copy=True).actor, nnx.Param
        )

        ec_opt_state = self.ec_optimizer.init(init_actor_params, ec_key)

        models = nnx.merge(agent_state.model_state, copy=True)
        opt_state = PyTreeDict(
            actor=nnx.unpack(
                nnx.Optimizer(
                    models.actor,
                    self.optimizer,
                    wrt=nnx.Param,
                )
            ),
            critic=nnx.unpack(
                nnx.Optimizer(
                    models.critic,
                    self.optimizer,
                    wrt=nnx.Param,
                )
            ),
        )

        return agent_state, opt_state, ec_opt_state

    # override
    def _rl_rollout(self, agent_state, replay_buffer_state, key):
        # agnet_state: only contains one agent
        # trajectory [T, B, ...]
        eval_metrics, trajectory = self.rl_collector.rollout(
            agent_state,
            key,
            self.config.rollout_episodes,
        )

        mask = jnp.logical_not(right_shift_with_padding(trajectory.dones, 1))
        trajectory = clean_trajectory(trajectory)
        trajectory, mask = tree_stop_gradient(
            flatten_rollout_trajectory((trajectory, mask))
        )
        replay_buffer_state = self.replay_buffer.add(
            replay_buffer_state, trajectory, mask
        )

        return eval_metrics, trajectory, replay_buffer_state

    # override
    def _rl_update(self, agent_state, opt_state, replay_buffer_state, key):
        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            key, rb_key, learn_key = jax.random.split(key, 3)

            rb_keys = jax.random.split(rb_key, self.config.actor_update_interval)
            # (actor_update_interval, B, ...)
            sample_batches = jax.vmap(self.replay_buffer.sample, in_axes=(None, 0))(
                replay_buffer_state, rb_keys
            )

            (agent_state, opt_state), train_info = update_erl_eda_td3(
                agent_state,
                opt_state,
                sample_batches,
                learn_key,
                agent=self.agent,
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

    def _rl_injection(self, ec_opt_state, agent_state):
        # update EC pop center with RL weights

        pop_mean = ec_opt_state.mean
        rl_actor_params = nnx.state(
            nnx.merge(agent_state.model_state, copy=True).actor, nnx.Param
        )

        # Tips: x = x + stepsize * (y - x)
        ec_opt_state = ec_opt_state.replace(
            mean=optax.incremental_update(
                rl_actor_params, pop_mean, self.config.rl_injection_stepsize
            )
        )

        return ec_opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        opt_state = state.opt_state
        ec_opt_state = state.ec_opt_state
        replay_buffer_state = state.replay_buffer_state
        iterations = state.metrics.iterations + 1

        key, ec_rollout_key, rl_rollout_key, learn_key = jax.random.split(
            state.key, num=4
        )

        # ======== EC rollout ========
        # the trajectory [#pop, T, B, ...]
        # metrics: [#pop, B]
        pop_actor_params, ec_opt_state = self.ec_optimizer.ask(ec_opt_state)

        pop_agent_state = make_erl_rollout_state(agent_state, pop_actor_params)
        ec_eval_metrics, ec_trajectory, replay_buffer_state = self._ec_rollout(
            pop_agent_state, replay_buffer_state, ec_rollout_key
        )

        ec_sampled_timesteps = ec_eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        ec_sampled_episodes = jnp.uint32(self.config.episodes_for_fitness * pop_size)

        # ======== RL update ========
        rl_eval_metrics, rl_trajectory, replay_buffer_state = self._rl_rollout(
            agent_state, replay_buffer_state, rl_rollout_key
        )

        rl_sampled_timesteps = rl_eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        rl_sampled_episodes = jnp.uint32(self.config.rollout_episodes)

        td3_metrics, agent_state, opt_state = self._rl_update(
            agent_state, opt_state, replay_buffer_state, learn_key
        )

        # ======== EC update ========
        fitnesses = ec_eval_metrics.episode_returns.mean(axis=-1)
        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        ec_opt_state = jax.lax.cond(
            iterations % self.config.rl_injection_interval == 0,
            self._rl_injection,
            lambda ec_opt_state, agent_state: ec_opt_state,
            ec_opt_state,
            agent_state,
        )

        train_metrics = ERLTrainMetric(
            pop_episode_lengths=ec_eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=ec_eval_metrics.episode_returns.mean(-1),
            rl_episode_lengths=rl_eval_metrics.episode_lengths.mean(-1),
            rl_episode_returns=rl_eval_metrics.episode_returns.mean(-1),
            rl_metrics=td3_metrics,
            ec_info=ec_metrics,
            rb_size=replay_buffer_state.buffer_size,
        )

        sampled_timesteps = ec_sampled_timesteps + rl_sampled_timesteps
        sampled_episodes = ec_sampled_episodes + rl_sampled_episodes
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
            rl_sampled_timesteps=state.metrics.rl_sampled_timesteps
            + rl_sampled_timesteps,
            iterations=iterations,
        )

        state = state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
            ec_opt_state=ec_opt_state,
            opt_state=opt_state,
        )

        return train_metrics, state

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        key, rl_eval_key, ec_eval_key = jax.random.split(state.key, num=3)

        rl_eval_metrics = self.evaluator.evaluate(
            state.agent_state, rl_eval_key, num_episodes=self.config.eval_episodes
        )

        pop_mean_actor_params = state.ec_opt_state.mean

        pop_mean_agent_state = make_erl_rollout_state(
            state.agent_state, pop_mean_actor_params
        )

        ec_eval_metrics = self.evaluator.evaluate(
            pop_mean_agent_state, ec_eval_key, num_episodes=self.config.eval_episodes
        )

        eval_metrics = EvaluateMetric(
            rl_episode_returns=rl_eval_metrics.episode_returns.mean(),
            rl_episode_lengths=rl_eval_metrics.episode_lengths.mean(),
            pop_center_episode_returns=ec_eval_metrics.episode_returns.mean(),
            pop_center_episode_lengths=ec_eval_metrics.episode_lengths.mean(),
        )

        state = state.replace(key=key)

        return eval_metrics, state

    def learn(self, state: State) -> State:
        sampled_episodes = int(state.metrics.sampled_episodes)
        start_iteration = int(state.metrics.iterations)
        sampled_episodes_per_iter = (
            self.config.episodes_for_fitness * self.config.pop_size
            + self.config.rollout_episodes
        )
        num_iters = math.ceil(
            (self.config.total_episodes - sampled_episodes) / sampled_episodes_per_iter
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

            self.recorder.write(train_metrics_dict, iters)

            if iters % self.config.eval_interval == 0 or iters == final_iteration:
                eval_metrics, state = self.evaluate(state)

                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iters
                )

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)

            self.checkpoint_manager.save(
                iters,
                saved_state,
                force=iters == final_iteration,
            )

        return state


def update_erl_eda_td3(
    agent_state: AgentState,
    opt_state: PyTreeDict,
    sample_batches,
    key: jax.Array,
    *,
    agent: Agent,
    config: DictConfig,
):
    """Update one TD3 learner used to guide an EDA population.

    Restore isolated NNX objects once, update them in place, then return snapshots.
    Forward-pass auxiliary variables are retained for the network being trained.
    Batches have axes `(critic_updates, batch, ...)`, with no population axis.
    """
    models = agent.get_models(agent_state, copy=True)
    actor_optimizer = nnx.merge(opt_state.actor, copy=True)
    critic_optimizer = nnx.merge(opt_state.critic, copy=True)

    def critic_loss_fn(models, sample_batch, key):
        losses = agent.critic_loss(
            agent_state.replace(model_state=nnx.unpack(models)), sample_batch, key
        )
        return losses.critic_loss, losses

    def actor_loss_fn(models, sample_batch, key):
        losses = agent.actor_loss(
            agent_state.replace(model_state=nnx.unpack(models)), sample_batch, key
        )
        return losses.actor_loss, losses

    def update_critic(models, optimizer, sample_batch, key):
        value, grads = nnx.value_and_grad(
            critic_loss_fn,
            argnums=nnx.DiffState(0, nnx.All(nnx.Param, nnx.PathContains("critic"))),
            has_aux=True,
        )(models, sample_batch, key)
        optimizer.update(models.critic, grads["critic"])
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
    actor_optimizer.update(models.actor, grads["actor"])
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
