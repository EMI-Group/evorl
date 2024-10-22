import logging
import math
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp

from evorl.replay_buffers import ReplayBuffer
from evorl.distributed import agent_gradient_update
from evorl.metrics import MetricBase, metricfield
from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import (
    tree_set,
    scan_and_mean,
    right_shift_with_padding,
    tree_stop_gradient,
)
from evorl.utils.rl_toolkits import soft_target_update, flatten_rollout_trajectory
from evorl.utils.ec_utils import flatten_pop_rollout_episode
from evorl.evaluators import Evaluator, EpisodeCollector
from evorl.agent import Agent, AgentState
from evorl.envs import create_env, AutoresetMode
from evorl.recorders import get_1d_array_statistics, add_prefix, get_1d_array
from evorl.ec.optimizers import ERLGA, ECState

from ..td3 import make_mlp_td3_agent, TD3NetworkParams, TD3TrainMetric
from ..offpolicy_utils import skip_replay_buffer_state
from .erl_utils import create_dummy_td3_trainmetric
from .erl_base import ERLWorkflowBase

logger = logging.getLogger(__name__)


class POPTrainMetric(MetricBase):
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array
    rb_size: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    rl_episode_returns: chex.Array | None = None
    rl_episode_lengths: chex.Array | None = None
    rl_metrics: MetricBase | None = None
    ec_info: PyTreeDict = metricfield(default_factory=PyTreeDict)


class EvaluateMetric(MetricBase):
    rl_episode_returns: chex.Array
    rl_episode_lengths: chex.Array


class ERLGAWorkflow(ERLWorkflowBase):
    """
    EC: n actors
    RL: k actors + k critics
    Shared replay buffer
    """

    @classmethod
    def name(cls):
        return "ERL-GA"

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

        ec_optimizer = ERLGA(
            pop_size=config.pop_size,
            num_elites=config.num_elites,
            weight_max_magnitude=config.weight_max_magnitude,
            mut_strength=config.mut_strength,
            num_mutation_frac=config.num_mutation_frac,
            super_mut_strength=config.super_mut_strength,
            super_mut_prob=config.super_mut_prob,
            reset_prob=config.reset_prob,
            vec_relative_prob=config.vec_relative_prob,
            enable_crossover=config.enable_crossover,
            num_crossover_frac=config.num_crossover_frac,
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
            config.env.env_name,
            config.env.env_type,
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
            obs_preprocessor_state=None,
        )

        workflow = cls(
            env,
            agent,
            agent_state_vmap_axes,
            optimizer,
            ec_optimizer,
            ec_collector,
            rl_collector,
            evaluator,
            replay_buffer,
            config,
        )

        workflow._rl_update_fn = build_rl_update_fn(
            agent, optimizer, config, agent_state_vmap_axes
        )

        return workflow

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree, ECState]:
        agent_key, pop_agent_key, ec_key = jax.random.split(key, 3)

        # agent for RL
        agent_state = jax.vmap(self.agent.init, in_axes=(None, None, 0))(
            self.env.obs_space,
            self.env.action_space,
            jax.random.split(agent_key, self.config.num_rl_agents),
        )

        # all agents will share the same obs_preprocessor_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=jtu.tree_map(
                    lambda x: x[0], agent_state.obs_preprocessor_state
                )
            )

        dummy_obs = self.env.obs_space.sample(key)
        pop_actor_params = jax.vmap(self.agent.actor_network.init, in_axes=(0, None))(
            jax.random.split(pop_agent_key, self.config.pop_size),
            dummy_obs[None, ...],
        )

        ec_opt_state = self.ec_optimizer.init(pop_actor_params, ec_key)

        opt_state = PyTreeDict(
            actor=self.optimizer.init(agent_state.params.actor_params),
            critic=self.optimizer.init(agent_state.params.critic_params),
        )

        return agent_state, opt_state, ec_opt_state

    def _rl_injection(
        self, ec_opt_state: ECState, agent_state: AgentState, fitnesses: chex.Array
    ) -> ECState:
        # replace EC worst individuals
        worst_indices = jax.lax.top_k(-fitnesses, self.config.num_rl_agents)[1]
        rl_actor_params = agent_state.params.actor_params

        chex.assert_tree_shape_prefix(rl_actor_params, (self.config.num_rl_agents,))

        ec_opt_state = ec_opt_state.replace(
            pop=tree_set(
                ec_opt_state.pop,
                rl_actor_params,
                worst_indices,
                unique_indices=True,
            )
        )

        return ec_opt_state

    def _ec_rollout(self, agent_state, replay_buffer_state, key):
        return self._rollout(
            agent_state, replay_buffer_state, key, self.config.pop_size
        )

    def _rl_rollout(self, agent_state, replay_buffer_state, key):
        return self._rollout(
            agent_state, replay_buffer_state, key, self.config.num_rl_agents
        )

    def _rollout(self, agent_state, replay_buffer_state, key, num_agents):
        eval_metrics, trajectory = jax.vmap(
            self.rl_collector.rollout,
            in_axes=(self.agent_state_vmap_axes, 0, None),
        )(
            agent_state,
            jax.random.split(key, num_agents),
            self.config.rollout_episodes,
        )

        # [n, T, B, ...] -> [T, n*B, ...]
        trajectory = trajectory.replace(next_obs=None)
        trajectory = flatten_pop_rollout_episode(trajectory)

        mask = jnp.logical_not(right_shift_with_padding(trajectory.dones, 1))
        trajectory = trajectory.replace(dones=None)
        trajectory, mask = tree_stop_gradient(
            flatten_rollout_trajectory((trajectory, mask))
        )
        replay_buffer_state = self.replay_buffer.add(
            replay_buffer_state, trajectory, mask
        )

        return eval_metrics, trajectory, replay_buffer_state

    def _rl_update(self, agent_state, opt_state, replay_buffer_state, key):
        def _sample_fn(key):
            return self.replay_buffer.sample(replay_buffer_state, key)

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            key, rb_key, learn_key = jax.random.split(key, 3)

            rb_keys = jax.random.split(
                rb_key, self.config.actor_update_interval * self.config.num_rl_agents
            )
            sample_batches = jax.vmap(_sample_fn)(rb_keys)

            # (actor_update_interval, num_rl_agents, B, ...)
            sample_batches = jax.tree_map(
                lambda x: x.reshape(
                    (
                        self.config.actor_update_interval,
                        self.config.num_rl_agents,
                        *x.shape[1:],
                    )
                ),
                sample_batches,
            )

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

        key, ec_rollout_key, rl_rollout_learn_key = jax.random.split(state.key, num=3)

        # ======== EC update ========
        # the trajectory [#pop, T, B, ...]
        # metrics: [#pop, B]
        pop_actor_params, ec_opt_state = self.ec_optimizer.ask(ec_opt_state)
        pop_agent_state = replace_td3_actor_params(agent_state, pop_actor_params)
        ec_eval_metrics, ec_trajectory, replay_buffer_state = self._ec_rollout(
            pop_agent_state, replay_buffer_state, ec_rollout_key
        )

        fitnesses = ec_eval_metrics.episode_returns.mean(axis=-1)
        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        # calculate the number of timestep
        sampled_timesteps += ec_eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        sampled_episodes += jnp.uint32(self.config.episodes_for_fitness * pop_size)

        train_metrics = POPTrainMetric(
            pop_episode_lengths=ec_eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=ec_eval_metrics.episode_returns.mean(-1),
            ec_info=ec_metrics,
        )

        # ======== RL update ========
        def _rl_rollout_and_update(
            agent_state, opt_state, replay_buffer_state, train_metrics, key
        ):
            rl_rollout_key, learn_key = jax.random.split(key, 2)
            rl_eval_metrics, rl_trajectory, replay_buffer_state = self._rl_rollout(
                agent_state, replay_buffer_state, rl_rollout_key
            )

            rl_sampled_timesteps = rl_eval_metrics.episode_lengths.sum().astype(
                jnp.uint32
            )
            rl_sampled_episodes = jnp.uint32(
                self.config.num_rl_agents * self.config.rollout_episodes
            )

            td3_metrics, agent_state, opt_state = self._rl_update(
                agent_state, opt_state, replay_buffer_state, learn_key
            )

            # get average loss
            td3_metrics = td3_metrics.replace(
                actor_loss=td3_metrics.actor_loss / self.config.num_rl_agents,
                critic_loss=td3_metrics.critic_loss / self.config.num_rl_agents,
            )

            train_metrics = train_metrics.replace(
                rl_episode_lengths=rl_eval_metrics.episode_lengths.mean(-1),
                rl_episode_returns=rl_eval_metrics.episode_returns.mean(-1),
                rl_metrics=td3_metrics,
            )

            return (
                train_metrics,
                rl_sampled_timesteps,
                rl_sampled_episodes,
                agent_state,
                opt_state,
                replay_buffer_state,
            )

        def _dummy_rl_rollout_and_update(
            agent_state, opt_state, replay_buffer_state, train_metrics, key
        ):
            train_metrics = train_metrics.replace(
                rl_episode_lengths=jnp.zeros((self.config.num_rl_agents,)),
                rl_episode_returns=jnp.zeros((self.config.num_rl_agents,)),
                rl_metrics=create_dummy_td3_trainmetric(self.config.num_rl_agents),
            )

            return (
                train_metrics,
                jnp.zeros((), dtype=jnp.uint32),
                jnp.zeros((), dtype=jnp.uint32),
                agent_state,
                opt_state,
                replay_buffer_state,
            )

        (
            train_metrics,
            rl_sampled_timesteps,
            rl_sampled_episodes,
            agent_state,
            opt_state,
            replay_buffer_state,
        ) = jax.lax.cond(
            iterations > self.config.warmup_iters,
            _rl_rollout_and_update,
            _dummy_rl_rollout_and_update,
            agent_state,
            opt_state,
            replay_buffer_state,
            train_metrics,
            rl_rollout_learn_key,
        )

        ec_opt_state = jax.lax.cond(
            jnp.logical_and(
                iterations > self.config.warmup_iters,
                iterations % self.config.rl_injection_interval == 0,
            ),
            self._rl_injection,
            lambda ec_opt_state, agent_state, fitnesses: ec_opt_state,
            ec_opt_state,
            agent_state,
            fitnesses,
        )

        sampled_timesteps += rl_sampled_timesteps
        sampled_episodes += rl_sampled_episodes
        train_metrics = train_metrics.replace(rb_size=replay_buffer_state.buffer_size)

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

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        key, eval_key = jax.random.split(state.key, num=2)

        # [num_rl_agents, #episodes]
        raw_eval_metrics = jax.vmap(
            self.evaluator.evaluate, in_axes=(self.agent_state_vmap_axes, 0, None)
        )(
            state.agent_state,
            jax.random.split(eval_key, self.config.num_rl_agents),
            self.config.eval_episodes,
        )

        eval_metrics = EvaluateMetric(
            rl_episode_returns=raw_eval_metrics.episode_returns.mean(-1),
            rl_episode_lengths=raw_eval_metrics.episode_lengths.mean(-1),
        )

        state = state.replace(key=key)
        return eval_metrics, state

    def learn(self, state: State) -> State:
        num_iters = math.ceil(
            (self.config.total_episodes - state.metrics.sampled_episodes)
            / (self.config.episodes_for_fitness * self.config.pop_size)
        )

        for i in range(state.metrics.iterations, num_iters + state.metrics.iterations):
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

            if self.config.num_rl_agents > 1:
                train_metrics_dict["rl_episode_lengths"] = get_1d_array_statistics(
                    train_metrics_dict["rl_episode_lengths"], histogram=True
                )
                train_metrics_dict["rl_episode_returns"] = get_1d_array_statistics(
                    train_metrics_dict["rl_episode_returns"], histogram=True
                )
                train_metrics_dict["rl_metrics"]["raw_loss_dict"] = jtu.tree_map(
                    get_1d_array_statistics,
                    train_metrics_dict["rl_metrics"]["raw_loss_dict"],
                )
            else:
                train_metrics_dict["rl_episode_lengths"] = train_metrics_dict[
                    "rl_episode_lengths"
                ].squeeze(0)
                train_metrics_dict["rl_episode_returns"] = train_metrics_dict[
                    "rl_episode_returns"
                ].squeeze(0)

            self.recorder.write(train_metrics_dict, iters)

            if iters % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)

                eval_metrics_dict = eval_metrics.to_local_dict()
                if self.config.num_rl_agents > 1:
                    eval_metrics_dict = jtu.tree_map(get_1d_array, eval_metrics_dict)

                self.recorder.write(add_prefix(eval_metrics_dict, "eval"), iters)

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)

            self.checkpoint_manager.save(iters, args=ocp.args.StandardSave(state))

        return state


def replace_td3_actor_params(
    agent_state: AgentState, pop_actor_params: chex.ArrayTree
) -> AgentState:
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


def build_rl_update_fn(
    agent: Agent,
    optimizer: optax.GradientTransformation,
    config: DictConfig,
    agent_state_vmap_axes: AgentState,
):
    num_rl_agents = config.num_rl_agents

    def critic_loss_fn(agent_state, sample_batch, key):
        # loss on a single critic with multiple actors
        # sample_batch: (n, B, ...)

        loss_dict = jax.vmap(agent.critic_loss, in_axes=(agent_state_vmap_axes, 0, 0))(
            agent_state, sample_batch, jax.random.split(key, num_rl_agents)
        )

        loss = loss_dict.critic_loss.sum()

        return loss, loss_dict

    def actor_loss_fn(agent_state, sample_batch, key):
        # loss on a single actor
        loss_dict = jax.vmap(agent.actor_loss, in_axes=(agent_state_vmap_axes, 0, 0))(
            agent_state, sample_batch, jax.random.split(key, num_rl_agents)
        )

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
