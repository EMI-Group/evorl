from flax import nnx
import logging
import math
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.metrics import MetricBase
from evorl.types import PyTreeDict, State
from evorl.agent import AgentState
from evorl.recorders import get_1d_array_statistics, add_prefix, get_1d_array
from evorl.ec.optimizers import ERLGAMod, ECState
from evorl.algorithms.offpolicy_utils import skip_replay_buffer_state

from .erl_workflow import (
    ERLTrainMetric,
    ERLWorkflowBase,
    make_erl_rollout_state,
)

logger = logging.getLogger(__name__)


class EvaluateMetric(MetricBase):
    rl_episode_returns: jax.Array
    rl_episode_lengths: jax.Array


class ERLGAWorkflow(ERLWorkflowBase):
    """ERL w/ GA.

    Config:

    - EC: n actors
    - RL: k actors + k critics
    - Shared replay buffer
    """

    @classmethod
    def name(cls):
        return "ERL-GA"

    @classmethod
    def _build_ec_optimizer(cls, config: DictConfig):
        assert config.num_elites >= config.num_rl_agents
        return ERLGAMod(
            pop_size=config.pop_size,
            num_elites=config.num_elites,
            external_size=config.num_rl_agents,
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

    def _setup_agent_and_optimizer(
        self, key: jax.Array
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

        pop_actor_params = jax.vmap(
            lambda k: nnx.state(
                nnx.merge(
                    self.agent.init(
                        self.env.obs_space, self.env.action_space, k
                    ).model_state,
                    copy=True,
                ).actor,
                nnx.Param,
            )
        )(jax.random.split(pop_agent_key, self.config.pop_size))

        ec_opt_state = self.ec_optimizer.init(pop_actor_params, ec_key)

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

        return agent_state, opt_state, ec_opt_state

    def _rl_injection(self, ec_opt_state: ECState, agent_state: AgentState) -> ECState:
        rl_actor_params = nnx.state(
            nnx.merge(agent_state.model_state, copy=True).actor, nnx.Param
        )
        chex.assert_tree_shape_prefix(rl_actor_params, (self.config.num_rl_agents,))

        ec_opt_state = ec_opt_state.replace(external_pop=rl_actor_params)

        return ec_opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        """The basic step function for the workflow to update agent."""
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

        # ====== EC update ======
        fitnesses = ec_eval_metrics.episode_returns.mean(axis=-1)

        def _tell_with_rl_injection(ec_opt_state, fitnesses):
            ec_opt_state = self._rl_injection(ec_opt_state, agent_state)
            ec_metrics, ec_opt_state = self.ec_optimizer.tell_external(
                ec_opt_state, fitnesses
            )
            return ec_metrics, ec_opt_state

        ec_metrics, ec_opt_state = jax.lax.cond(
            iterations % self.config.rl_injection_interval == 0,
            _tell_with_rl_injection,
            self.ec_optimizer.tell,
            ec_opt_state,
            fitnesses,
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

        # calculate the number of timestep
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
        sampled_episodes = int(state.metrics.sampled_episodes)
        start_iteration = int(state.metrics.iterations)
        sampled_episodes_per_iter = (
            self.config.episodes_for_fitness * self.config.pop_size
            + self.config.rollout_episodes * self.config.num_rl_agents
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

            if iters % self.config.eval_interval == 0 or iters == final_iteration:
                eval_metrics, state = self.evaluate(state)

                eval_metrics_dict = eval_metrics.to_local_dict()
                if self.config.num_rl_agents > 1:
                    eval_metrics_dict = jtu.tree_map(get_1d_array, eval_metrics_dict)

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
