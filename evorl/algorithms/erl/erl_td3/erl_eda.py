import logging
import math
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.replay_buffers import ReplayBuffer
from evorl.distributed import agent_gradient_update
from evorl.metrics import MetricBase
from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import (
    tree_stop_gradient,
    rng_split_like_tree,
    right_shift_with_padding,
    scan_and_mean,
)
from evorl.utils.rl_toolkits import soft_target_update, flatten_rollout_trajectory
from evorl.evaluators import Evaluator, EpisodeCollector
from evorl.agent import AgentState, Agent
from evorl.envs import create_env, AutoresetMode
from evorl.recorders import get_1d_array_statistics, add_prefix
from evorl.ec.optimizers import ECState, OpenES, ExponentialScheduleSpec
from evorl.algorithms.td3 import make_mlp_td3_agent, TD3TrainMetric
from evorl.algorithms.offpolicy_utils import clean_trajectory, skip_replay_buffer_state

from ..erl_workflow import ERLTrainMetric
from .erl_td3_workflow import erl_replace_td3_actor_params, ERLTD3WorkflowTemplate

logger = logging.getLogger(__name__)


class EvaluateMetric(MetricBase):
    rl_episode_returns: chex.Array
    rl_episode_lengths: chex.Array
    pop_center_episode_returns: chex.Array
    pop_center_episode_lengths: chex.Array


class ERLEDAWorkflow(ERLTD3WorkflowTemplate):
    """ERL w/ EDA.

    Configs:

    - EC: n actors
    - RL: 1 (actor,critic)
    - Shared replay buffer

    RL will be injected into the pop mean. Support all EDA based ES algorithms.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # override
        self._rl_update_fn = build_rl_update_fn(self.agent, self.optimizer, self.config)

    @classmethod
    def name(cls):
        return "ERL-EDA"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        # env for rl&ec rollout
        env = create_env(
            config.env,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
            record_ori_obs=True,
        )

        agent = make_mlp_td3_agent(
            action_space=env.action_space,
            norm_layer_type=config.agent_network.norm_layer_type,
            num_critics=config.agent_network.num_critics,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            critics_in_actor_loss=config.critics_in_actor_loss,
            normalize_obs=config.normalize_obs,
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

        ec_optimizer = OpenES(
            pop_size=config.pop_size,
            lr_schedule=ExponentialScheduleSpec(**config.ec_lr),
            noise_std_schedule=ExponentialScheduleSpec(**config.ec_noise_std),
            mirror_sampling=config.mirror_sampling,
        )

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
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        # this is only used for _ec_rollout()
        agent_state_vmap_axes = AgentState(
            params=0,
            obs_preprocessor_state=None,
        )

        workflow = cls(
            env=env,
            agent=agent,
            agent_state_vmap_axes=agent_state_vmap_axes,
            optimizer=optimizer,
            ec_optimizer=ec_optimizer,
            ec_collector=ec_collector,
            rl_collector=rl_collector,
            evaluator=evaluator,
            replay_buffer=replay_buffer,
            config=config,
        )

        return workflow

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree, ECState]:
        agent_key, ec_key = jax.random.split(key)

        # one agent for RL
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )

        init_actor_params = agent_state.params.actor_params

        ec_opt_state = self.ec_optimizer.init(init_actor_params, ec_key)

        opt_state = PyTreeDict(
            actor=self.optimizer.init(agent_state.params.actor_params),
            critic=self.optimizer.init(agent_state.params.critic_params),
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
        def _sample_fn(key):
            return self.replay_buffer.sample(replay_buffer_state, key)

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            key, rb_key, learn_key = jax.random.split(key, 3)

            rb_keys = jax.random.split(rb_key, self.config.actor_update_interval)
            # (actor_update_interval, B, ...)
            sample_batches = jax.vmap(_sample_fn)(rb_keys)

            (agent_state, opt_state), train_info = self._rl_update_fn(
                agent_state, opt_state, sample_batches, learn_key
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
        rl_actor_params = agent_state.params.actor_params

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

        if self.config.mirror_sampling:
            key, perm_key = jax.random.split(key)
            pop_actor_params = jtu.tree_map(
                lambda x, k: jax.random.permutation(k, x, axis=0),
                pop_actor_params,
                rng_split_like_tree(perm_key, pop_actor_params),
            )

        pop_agent_state = erl_replace_td3_actor_params(agent_state, pop_actor_params)
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

        sampled_timesteps = ec_sampled_episodes + rl_sampled_timesteps
        sampled_episodes = ec_sampled_timesteps + rl_sampled_episodes
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

        pop_mean_agent_state = erl_replace_td3_actor_params(
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
        sampled_episodes_per_iter = (
            self.config.episodes_for_fitness * self.config.pop_size
            + self.config.rollout_episodes
        )
        num_iters = math.ceil(
            (self.config.total_episodes - state.metrics.sampled_episodes)
            / sampled_episodes_per_iter
        )

        final_iteration = num_iters + state.metrics.iterations
        for i in range(state.metrics.iterations, final_iteration):
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


def build_rl_update_fn(
    agent: Agent,
    optimizer: optax.GradientTransformation,
    config: DictConfig,
):
    def critic_loss_fn(agent_state, sample_batch, key):
        # loss on a single critic with multiple actors
        # sample_batch: (B, ...)

        loss_dict = agent.critic_loss(agent_state, sample_batch, key)

        loss = loss_dict.critic_loss

        return loss, loss_dict

    def actor_loss_fn(agent_state, sample_batch, key):
        # loss on a single actor
        # different actor shares same sample_batch (B, ...) input
        loss_dict = agent.actor_loss(agent_state, sample_batch, key)

        loss = loss_dict.actor_loss

        return loss, loss_dict

    critic_update_fn = agent_gradient_update(
        critic_loss_fn,
        optimizer,
        has_aux=True,
        attach_fn=lambda agent_state, critic_params: agent_state.replace(
            params=agent_state.params.replace(critic_params=critic_params)
        ),
        detach_fn=lambda agent_state: agent_state.params.critic_params,
    )

    actor_update_fn = agent_gradient_update(
        actor_loss_fn,
        optimizer,
        has_aux=True,
        attach_fn=lambda agent_state, actor_params: agent_state.replace(
            params=agent_state.params.replace(actor_params=actor_params)
        ),
        detach_fn=lambda agent_state: agent_state.params.actor_params,
    )

    def _update_fn(agent_state, opt_state, sample_batches, key):
        critic_opt_state = opt_state.critic
        actor_opt_state = opt_state.actor

        key, critic_key, actor_key = jax.random.split(key, num=3)

        critic_sample_batches = jtu.tree_map(lambda x: x[:-1], sample_batches)
        last_sample_batch = jtu.tree_map(lambda x: x[-1], sample_batches)

        if config.actor_update_interval - 1 > 0:

            def _update_critic_fn(carry, sample_batch):
                key, agent_state, critic_opt_state = carry

                key, critic_key = jax.random.split(key)

                (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
                    critic_update_fn(
                        critic_opt_state, agent_state, sample_batch, critic_key
                    )
                )

                return (key, agent_state, critic_opt_state), None

            key, critic_multiple_update_key = jax.random.split(key)

            (_, agent_state, critic_opt_state), _ = jax.lax.scan(
                _update_critic_fn,
                (
                    critic_multiple_update_key,
                    agent_state,
                    critic_opt_state,
                ),
                critic_sample_batches,
                length=config.actor_update_interval - 1,
            )

        (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
            critic_update_fn(
                critic_opt_state, agent_state, last_sample_batch, critic_key
            )
        )

        (actor_loss, actor_loss_dict), agent_state, actor_opt_state = actor_update_fn(
            actor_opt_state, agent_state, last_sample_batch, actor_key
        )

        # not need vmap
        target_actor_params = soft_target_update(
            agent_state.params.target_actor_params,
            agent_state.params.actor_params,
            config.tau,
        )
        target_critic_params = soft_target_update(
            agent_state.params.target_critic_params,
            agent_state.params.critic_params,
            config.tau,
        )
        agent_state = agent_state.replace(
            params=agent_state.params.replace(
                target_actor_params=target_actor_params,
                target_critic_params=target_critic_params,
            )
        )

        opt_state = opt_state.replace(actor=actor_opt_state, critic=critic_opt_state)

        return (
            (agent_state, opt_state),
            (critic_loss, actor_loss, critic_loss_dict, actor_loss_dict),
        )

    return _update_fn
