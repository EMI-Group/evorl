from functools import partial
import math
from typing_extensions import Self  # pytype: disable=not-supported-yet]
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.distributed import agent_gradient_update
from evorl.agent import Agent, AgentState
from evorl.types import PyTreeDict, State
from evorl.metrics import MetricBase, EvaluateMetric
from evorl.replay_buffers import ReplayBuffer
from evorl.envs import create_env, AutoresetMode
from evorl.utils.rl_toolkits import (
    soft_target_update,
)
from evorl.recorders import add_prefix, get_1d_array_statistics, get_1d_array
from evorl.evaluators import Evaluator, EpisodeCollector


from evorl.algorithms.offpolicy_utils import skip_replay_buffer_state
from evorl.algorithms.td3 import make_mlp_td3_agent
from evorl.algorithms.erl.cemrl_td3.cemrl_td3_workflow import CEMRLTD3WorkflowTemplate
from evorl.algorithms.erl.cemrl_workflow import CEMRLTrainMetric


class PopEpisodicTD3Workflow(CEMRLTD3WorkflowTemplate):
    """A batched TD3 workflow like CEMERL.

    The differences from CEMRL are:
    - Each individual has an actor and a critic.
    - All individuals are updated by RL.
    """

    def __init__(self, **kwargs):
        super(CEMRLTD3WorkflowTemplate, self).__init__(**kwargs)
        self._rl_update_fn = build_rl_update_fn(
            self.agent,
            self.optimizer,
            self.config,
            self.agent_state_vmap_axes,
        )

    @classmethod
    def name(cls):
        return "PopEpisodicTD3"

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        assert config.random_timesteps > 0, (
            "random_timesteps should be positive to pre-fill some data in the replay buffer"
        )

        assert config.pop_size == config.num_learning_offspring, (
            "pop_size must equal to num_learning_offspring"
        )

        # env for one actor
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
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        agent_state_vmap_axes = AgentState(
            params=0,
            obs_preprocessor_state=None,  # shared
        )

        workflow = cls(
            env=env,
            agent=agent,
            agent_state_vmap_axes=agent_state_vmap_axes,
            optimizer=optimizer,
            ec_optimizer=None,
            collector=collector,
            evaluator=evaluator,
            replay_buffer=replay_buffer,
            config=config,
        )

        return workflow

    def _setup_agent_and_optimizer(self, key: chex.PRNGKey):
        agent_state = jax.vmap(self.agent.init, in_axes=(None, None, 0))(
            self.env.obs_space,
            self.env.action_space,
            jax.random.split(key, self.config.pop_size),
        )

        opt_state = PyTreeDict(
            actor=self.optimizer.init(agent_state.params.actor_params),
            critic=self.optimizer.init(agent_state.params.critic_params),
        )

        ec_opt_state = None

        return agent_state, opt_state, ec_opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        opt_state = state.opt_state
        ec_opt_state = state.ec_opt_state
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

        train_metrics = CEMRLTrainMetric(
            rb_size=replay_buffer_state.buffer_size,
            pop_episode_lengths=eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=eval_metrics.episode_returns.mean(-1),
            rl_metrics=td3_metrics,
        )

        # calculate the number of timestep
        sampled_timesteps = eval_metrics.episode_lengths.sum().astype(jnp.uint32)
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
            ec_opt_state=ec_opt_state,
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
            episode_returns=raw_eval_metrics.episode_returns.mean(-1),
            episode_lengths=raw_eval_metrics.episode_lengths.mean(-1),
        )

        state = state.replace(key=key)
        return eval_metrics, state

    def learn(self, state: State) -> State:
        num_iters = math.ceil(
            (self.config.total_episodes - state.metrics.sampled_episodes)
            / (self.config.episodes_for_fitness * self.config.pop_size)
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


def build_rl_update_fn(
    agent: Agent,
    optimizer: optax.GradientTransformation,
    config: DictConfig,
    agent_state_vmap_axes: AgentState,
):
    """K actors + 1 shared critic."""
    num_learning_offspring = config.num_learning_offspring

    def critic_loss_fn(agent_state, sample_batch, key):
        # loss on a single critic with multiple actors
        # sample_batch: (n, B, ...)

        loss_dict = jax.vmap(agent.critic_loss, in_axes=(agent_state_vmap_axes, 0, 0))(
            agent_state, sample_batch, jax.random.split(key, num_learning_offspring)
        )

        # ++++++++++ diff +++++++++
        loss = loss_dict.critic_loss.sum()
        # +++++++++++++++++++++++++

        return loss, loss_dict

    def actor_loss_fn(agent_state, sample_batch, key):
        # loss on a single actor

        loss_dict = jax.vmap(agent.actor_loss, in_axes=(agent_state_vmap_axes, 0, 0))(
            agent_state, sample_batch, jax.random.split(key, num_learning_offspring)
        )

        # sum over the num_learning_offspring
        loss = loss_dict.actor_loss.sum()

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
