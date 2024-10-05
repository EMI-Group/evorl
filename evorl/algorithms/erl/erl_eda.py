import logging
import math
from omegaconf import DictConfig

import chex
import flashbax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from evorl.distributed import agent_gradient_update
from evorl.metrics import MetricBase, metricfield
from evorl.types import (
    PyTreeDict,
    State,
)
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.rl_toolkits import (
    soft_target_update,
)
from evorl.utils.flashbax_utils import get_buffer_size
from evorl.evaluator import Evaluator
from evorl.agent import AgentState, Agent
from evorl.envs import create_env, AutoresetMode, Box
from evorl.recorders import get_1d_array_statistics, add_prefix
from evorl.ec.optimizers import ECState, OpenES, ExponetialScheduleSpec

from ..td3 import TD3Agent, TD3NetworkParams
from ..offpolicy_utils import clean_trajectory, skip_replay_buffer_state
from .episode_collector import EpisodeCollector
from .erl_ga import ERLGAWorkflow

logger = logging.getLogger(__name__)


class POPTrainMetric(MetricBase):
    rb_size: int
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array
    rl_episode_returns: chex.Array | None = None
    rl_episode_lengths: chex.Array | None = None
    rl_metrics: MetricBase | None = None
    ec_info: PyTreeDict = metricfield(default_factory=PyTreeDict)


class WorkflowMetric(MetricBase):
    sampled_timesteps: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    rl_sampled_timesteps: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class EvaluateMetric(MetricBase):
    rl_episode_returns: chex.Array
    rl_episode_lengths: chex.Array
    pop_center_episode_returns: chex.Array
    pop_center_episode_lengths: chex.Array


class ERLEDAWorkflow(ERLGAWorkflow):
    """
    EC: n actors
    RL: k actors + k critics + 1 replay buffer.
    """

    @classmethod
    def name(cls):
        return "ERL-EDA"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        """
        return workflow
        """

        # env for rl&ec rollout
        env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
            record_ori_obs=True,
        )

        assert isinstance(
            env.action_space, Box
        ), "Only continue action space is supported."

        agent = TD3Agent(
            num_critics=config.agent_network.num_critics,
            norm_layer_type=config.agent_network.norm_layer_type,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            critics_in_actor_loss=config.critics_in_actor_loss,
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
            lr_schedule=ExponetialScheduleSpec(**config.ec_optimizer.lr),
            noise_stdev_schedule=ExponetialScheduleSpec(
                **config.ec_optimizer.noise_stdev
            ),
            mirror_sampling=config.mirror_sampling,
        )

        if config.fitness_with_exploration:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        ec_collector = EpisodeCollector(
            env_step_fn=env.step,
            env_reset_fn=env.reset,
            action_fn=action_fn,
            num_envs=config.num_envs,
            max_episode_steps=config.env.max_episode_steps,
            env_extra_fields=("ori_obs", "termination"),
        )

        if config.rl_exploration:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        rl_collector = EpisodeCollector(
            env_step_fn=env.step,
            env_reset_fn=env.reset,
            action_fn=action_fn,
            num_envs=config.num_envs,
            max_episode_steps=config.env.max_episode_steps,
            env_extra_fields=("ori_obs", "termination"),
        )

        replay_buffer = flashbax.make_item_buffer(
            max_length=config.replay_buffer_capacity,
            min_length=config.batch_size,
            sample_batch_size=config.batch_size,
            add_batches=True,
        )

        # to evaluate the pop-mean actor
        eval_env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            agent=agent,
            max_episode_steps=config.env.max_episode_steps,
        )

        workflow = cls(
            env,
            agent,
            optimizer,
            ec_optimizer,
            ec_collector,
            rl_collector,
            evaluator,
            replay_buffer,
            config,
        )

        workflow.agent_state_pytree_axes = AgentState(
            params=0,
            obs_preprocessor_state=None,
        )

        workflow._rl_update_fn = _build_rl_update_fn(
            agent, optimizer, config, workflow.agent_state_pytree_axes
        )

        return workflow

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree, ECState]:
        agent_key = key

        # one agent for RL
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )

        init_actor_params = agent_state.params.actor_params

        ec_opt_state = self.ec_optimizer.init(init_actor_params)

        opt_state = PyTreeDict(
            actor=self.optimizer.init(agent_state.params.actor_params),
            critic=self.optimizer.init(agent_state.params.critic_params),
        )

        return agent_state, opt_state, ec_opt_state

    def _rl_rollout(self, agent_state, key):
        eval_metrics, trajectory = self.rl_collector.evaluate(
            agent_state,
            self.config.rollout_episodes,
            key,
        )

        trajectory = clean_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        return eval_metrics, trajectory

    def step(self, state: State) -> tuple[MetricBase, State]:
        """
        the basic step function for the workflow to update agent
        """
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        opt_state = state.opt_state
        ec_opt_state = state.ec_opt_state
        replay_buffer_state = state.replay_buffer_state
        iterations = state.metrics.iterations + 1

        sampled_timesteps = jnp.zeros((), dtype=jnp.uint32)
        sampled_episodes = jnp.zeros((), dtype=jnp.uint32)

        key, rb_sample_key, ec_rollout_key, rl_rollout_key, ec_key, learn_key = (
            jax.random.split(state.key, num=6)
        )

        # ======== EC update ========
        # the trajectory [#pop, T, B, ...]
        # metrics: [#pop, B]

        pop_actor_params = self._ec_generate(ec_opt_state, ec_key)
        pop_agent_state = replace_actor_params(agent_state, pop_actor_params)
        ec_eval_metrics, ec_trajectory = self._ec_rollout(
            pop_agent_state, ec_rollout_key
        )

        fitnesses = ec_eval_metrics.episode_returns.mean(axis=-1)

        replay_buffer_state = self._add_to_replay_buffer(
            replay_buffer_state,
            ec_trajectory,
            ec_eval_metrics.episode_lengths.flatten(),
        )

        ec_opt_state = self._ec_update(ec_opt_state, pop_actor_params, fitnesses)

        # calculate the number of timestep
        sampled_timesteps += ec_eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        sampled_episodes += jnp.uint32(self.config.episodes_for_fitness * pop_size)

        train_metrics = POPTrainMetric(
            rb_size=0,
            pop_episode_lengths=ec_eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=ec_eval_metrics.episode_returns.mean(-1),
        )

        # ======== RL update ========
        if iterations > self.config.warmup_iters:
            # RL agent rollout with action noise
            rl_eval_metrics, rl_trajectory = self._rl_rollout(
                agent_state, rl_rollout_key
            )

            replay_buffer_state = self._add_to_replay_buffer(
                replay_buffer_state,
                rl_trajectory,
                rl_eval_metrics.episode_lengths.flatten(),
            )

            td3_metrics, agent_state, opt_state = self._rl_update(
                agent_state, opt_state, replay_buffer_state, learn_key
            )

            rl_sampled_timesteps = rl_eval_metrics.episode_lengths.sum().astype(
                jnp.uint32
            )
            sampled_timesteps += rl_sampled_timesteps
            sampled_episodes += jnp.uint32(self.config.rollout_episodes)

            if iterations % self.config.rl_injection_interval == 0:
                # replace the center
                ec_opt_state = self._rl_injection(agent_state, ec_opt_state)

            train_metrics = train_metrics.replace(
                rl_episode_lengths=rl_eval_metrics.episode_lengths.mean(-1),
                rl_episode_returns=rl_eval_metrics.episode_returns.mean(-1),
                rl_metrics=td3_metrics,
            )

        else:
            rl_sampled_timesteps = jnp.zeros((), dtype=jnp.uint32)

        train_metrics = train_metrics.replace(
            rb_size=get_buffer_size(replay_buffer_state),
        )

        # iterations is the number of updates of the agent
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

    def _ec_generate(self, ec_opt_state, key):
        return self.ec_optimizer.ask(ec_opt_state, key)

    def _rl_injection(self, agent_state, ec_opt_state):
        # update EC pop center with RL weights

        pop_mean = ec_opt_state.mean
        rl_actor_params = agent_state.params.actor_params

        ec_opt_state = ec_opt_state.replace(
            mean=optax.incremental_update(
                rl_actor_params, pop_mean, self.config.rl_injection_update_stepsize
            )
        )

        return ec_opt_state

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        key, rl_eval_key, ec_eval_key = jax.random.split(state.key, num=3)

        rl_eval_metrics = self.evaluator.evaluate(
            state.agent_state, rl_eval_key, num_episodes=self.config.eval_episodes
        )

        pop_mean_actor_params = state.ec_opt_state.mean

        pop_mean_agent_state = replace_actor_params(
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
        num_iters = math.ceil(
            self.config.total_episodes
            / (self.config.episodes_for_fitness * self.config.pop_size)
        )

        for i in range(state.metrics.iterations, num_iters):
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

            if iters % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)

                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iters
                )

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)

            self.checkpoint_manager.save(iters, args=ocp.args.StandardSave(state))

        return state


def replace_actor_params(agent_state: AgentState, pop_actor_params) -> AgentState:
    """
    reset the actor params and target actor params
    """

    return agent_state.replace(
        params=TD3NetworkParams(
            actor_params=pop_actor_params,
            target_actor_params=pop_actor_params,
            critic_params=None,
            target_critic_params=None,
        )
    )


def _build_rl_update_fn(
    agent: Agent,
    optimizer: optax.GradientTransformation,
    config: DictConfig,
    agent_state_pytree_axes: AgentState,
):
    def critic_loss_fn(agent_state, sample_batch, key):
        # loss on a single critic with multiple actors
        # sample_batch: (B, ...)

        loss_dict = agent.critic_loss(agent_state, sample_batch, key)

        loss = config.loss_weights.critic_loss * loss_dict.critic_loss.sum()

        return loss, loss_dict

    def actor_loss_fn(agent_state, sample_batch, key):
        # loss on a single actor
        # different actor shares same sample_batch (B, ...) input
        loss_dict = agent.actor_loss(agent_state, sample_batch, key)

        loss = config.loss_weights.actor_loss * loss_dict.actor_loss.sum()

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

        critic_sample_batches = jax.tree_map(lambda x: x[:-1], sample_batches)
        last_sample_batch = jax.tree_map(lambda x: x[-1], sample_batches)

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
