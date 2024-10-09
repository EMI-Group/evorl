import logging
import math
from functools import partial
from omegaconf import DictConfig

import chex
import flashbax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp

from evorl.distributed import agent_gradient_update
from evorl.metrics import MetricBase
from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import tree_set
from evorl.utils.rl_toolkits import soft_target_update
from evorl.utils.flashbax_utils import get_buffer_size
from evorl.evaluator import Evaluator
from evorl.agent import Agent, AgentState
from evorl.envs import create_env, AutoresetMode
from evorl.recorders import get_1d_array_statistics, add_prefix
from evorl.ec.optimizers import ERLGA, ECState

from ..td3 import make_mlp_td3_agent, TD3NetworkParams
from ..offpolicy_utils import skip_replay_buffer_state
from .episode_collector import EpisodeCollector
from .erl_base import ERLWorkflowTemplate, POPTrainMetric

logger = logging.getLogger(__name__)


class ERLGAWorkflow(ERLWorkflowTemplate):
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

        agent_state_pytree_axes = AgentState(
            params=0,
            obs_preprocessor_state=None,
        )

        workflow = cls(
            env,
            agent,
            agent_state_pytree_axes,
            optimizer,
            ec_optimizer,
            ec_collector,
            rl_collector,
            evaluator,
            replay_buffer,
            config,
        )

        workflow._rl_update_fn = build_rl_update_fn(
            agent, optimizer, config, workflow.agent_state_pytree_axes
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

        pop_actor_params = jax.vmap(self.agent.actor_network.init, in_axes=(0, None))(
            jax.random.split(pop_agent_key, self.config.pop_size),
            jnp.zeros((1, self.env.obs_space.shape[0])),
        )

        ec_opt_state = self.ec_optimizer.init(pop_actor_params, ec_key)

        opt_state = PyTreeDict(
            actor=jax.vmap(self.optimizer.init)(agent_state.params.actor_params),
            critic=jax.vmap(self.optimizer.init)(agent_state.params.critic_params),
        )

        return agent_state, opt_state, ec_opt_state

    def _rl_injection(
        self, ec_opt_state: ECState, agent_state: AgentState, fitnesses: chex.Array
    ) -> ECState:
        # replace EC worst individuals
        worst_indices = fitnesses.argsort()[: self.config.num_rl_agents]
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

        key, ec_rollout_key, rl_rollout_key, learn_key = jax.random.split(
            state.key, num=4
        )

        # ======== EC update ========
        # the trajectory [#pop, T, B, ...]
        # metrics: [#pop, B]
        pop_actor_params = ec_opt_state.pop
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
            pop_episode_lengths=ec_eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=ec_eval_metrics.episode_returns.mean(-1),
        )

        # ======== RL update ========
        if iterations > self.config.warmup_iters:
            rl_eval_metrics, rl_trajectory = self._rl_rollout(
                agent_state, rl_rollout_key
            )

            replay_buffer_state = self._add_to_replay_buffer(
                replay_buffer_state,
                rl_trajectory,
                rl_eval_metrics.episode_lengths.flatten(),
            )

            rl_sampled_timesteps = rl_eval_metrics.episode_lengths.sum()
            sampled_timesteps += rl_sampled_timesteps.astype(jnp.uint32)
            sampled_episodes += jnp.uint32(
                self.config.num_rl_agents * self.config.rollout_episodes
            )

            td3_metrics, agent_state, opt_state = self._rl_update(
                agent_state, opt_state, replay_buffer_state, learn_key
            )

            if iterations % self.config.rl_injection_interval == 0:
                ec_opt_state = self._rl_injection(ec_opt_state, agent_state, fitnesses)

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

            if train_metrics_dict["rl_metrics"] is not None:
                if self.config.num_rl_agents > 1:
                    train_metrics_dict["rl_episode_lengths"] = get_1d_array_statistics(
                        train_metrics_dict["rl_episode_lengths"], histogram=True
                    )
                    train_metrics_dict["rl_episode_returns"] = get_1d_array_statistics(
                        train_metrics_dict["rl_episode_returns"], histogram=True
                    )

                    train_metrics_dict["rl_metrics"]["actor_loss"] /= (
                        self.config.num_rl_agents
                    )
                    train_metrics_dict["rl_metrics"]["critic_loss"] /= (
                        self.config.num_rl_agents
                    )
                    train_metrics_dict["rl_metrics"]["raw_loss_dict"] = jtu.tree_map(
                        get_1d_array_statistics,
                        train_metrics_dict["rl_metrics"]["raw_loss_dict"],
                    )
                else:
                    train_metrics_dict["rl_episode_lengths"] = train_metrics_dict[
                        "rl_episode_lengths"
                    ].squeeze(-1)
                    train_metrics_dict["rl_episode_returns"] = train_metrics_dict[
                        "rl_episode_returns"
                    ].squeeze(-1)

            self.recorder.write(train_metrics_dict, iters)

            if iters % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)

                eval_metrics_dict = eval_metrics.to_local_dict()
                if self.config.num_rl_agents > 1:
                    eval_metrics_dict = jtu.tree_map(
                        partial(get_1d_array_statistics, histogram=True),
                        eval_metrics_dict,
                    )

                self.recorder.write(add_prefix(eval_metrics_dict, "eval"), iters)

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


def build_rl_update_fn(
    agent: Agent,
    optimizer: optax.GradientTransformation,
    config: DictConfig,
    agent_state_pytree_axes: AgentState,
):
    num_rl_agents = config.num_rl_agents

    def critic_loss_fn(agent_state, sample_batch, key):
        # loss on a single critic with multiple actors
        # sample_batch: (n, B, ...)

        loss_dict = jax.vmap(
            agent.critic_loss, in_axes=(agent_state_pytree_axes, 0, 0)
        )(agent_state, sample_batch, jax.random.split(key, num_rl_agents))

        loss = loss_dict.critic_loss.sum()

        return loss, loss_dict

    def actor_loss_fn(agent_state, sample_batch, key):
        # loss on a single actor
        loss_dict = jax.vmap(agent.actor_loss, in_axes=(agent_state_pytree_axes, 0, 0))(
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
