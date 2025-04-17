import logging
import math
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.replay_buffers import ReplayBuffer
from evorl.metrics import MetricBase
from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import tree_get
from evorl.evaluators import Evaluator, EpisodeCollector
from evorl.agent import AgentState
from evorl.envs import create_env, AutoresetMode
from evorl.recorders import get_1d_array_statistics, add_prefix, get_1d_array
from evorl.ec.optimizers import ECState, VanillaESMod, ExponentialScheduleSpec
from evorl.algorithms.td3 import make_mlp_td3_agent
from evorl.algorithms.offpolicy_utils import skip_replay_buffer_state

from ..erl_workflow import ERLTrainMetric
from .erl_td3_workflow import ERLTD3WorkflowTemplate, erl_replace_td3_actor_params

logger = logging.getLogger(__name__)


class EvaluateMetric(MetricBase):
    rl_episode_returns: chex.Array
    rl_episode_lengths: chex.Array
    pop_center_episode_returns: chex.Array
    pop_center_episode_lengths: chex.Array


class ERLESWorkflow(ERLTD3WorkflowTemplate):
    """ERL w/ ES.

    Configs:

    - EC: n actors
    - RL: k actors + k critics
    - Shared replay buffer
    """

    @classmethod
    def name(cls):
        return "ERL-ES"

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

        ec_optimizer = VanillaESMod(
            pop_size=config.pop_size,
            external_size=config.num_rl_agents,
            num_elites=config.num_elites,
            noise_std_schedule=ExponentialScheduleSpec(**config.ec_noise_std),
            mix_strategy=config.mix_strategy,
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
                obs_preprocessor_state=tree_get(agent_state.obs_preprocessor_state, 0)
            )

        dummy_obs = self.env.obs_space.sample(key)
        init_actor_params = self.agent.actor_network.init(
            pop_agent_key, jtu.tree_map(lambda x: x[None, ...], dummy_obs)
        )

        ec_opt_state = self.ec_optimizer.init(init_actor_params, ec_key)

        opt_state = PyTreeDict(
            actor=self.optimizer.init(agent_state.params.actor_params),
            critic=self.optimizer.init(agent_state.params.critic_params),
        )

        return agent_state, opt_state, ec_opt_state

    def _rl_injection(
        self,
        ec_opt_state: ECState,
        agent_state: AgentState,
        ec_fitnesses: chex.Array,
        rl_fitnesses: chex.Array,
    ) -> tuple[chex.Array, ECState]:
        rl_noise = jtu.tree_map(
            lambda x, m: x - m,
            agent_state.params.actor_params,
            ec_opt_state.mean,
        )

        concat_noise = jtu.tree_map(
            lambda n1, n2: jnp.concatenate([n1, n2], axis=0),
            ec_opt_state.noise,
            rl_noise,
        )

        ec_opt_state = ec_opt_state.replace(noise=concat_noise)

        fitnesses = jnp.concatenate([ec_fitnesses, rl_fitnesses], axis=0)

        return fitnesses, ec_opt_state

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

        # ======== EC & RL rollout ========
        # the trajectory [#pop, T, B, ...]
        # metrics: [#pop, B]
        pop_actor_params, ec_opt_state = self.ec_optimizer.ask(ec_opt_state)

        pop_agent_state = erl_replace_td3_actor_params(agent_state, pop_actor_params)
        ec_eval_metrics, ec_trajectory, replay_buffer_state = self._ec_rollout(
            pop_agent_state, replay_buffer_state, ec_rollout_key
        )

        ec_sampled_timesteps = ec_eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        ec_sampled_episodes = jnp.uint32(self.config.episodes_for_fitness * pop_size)

        rl_eval_metrics, rl_trajectory, replay_buffer_state = self._rl_rollout(
            agent_state, replay_buffer_state, rl_rollout_key
        )

        rl_sampled_timesteps = rl_eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        rl_sampled_episodes = jnp.uint32(
            self.config.num_rl_agents * self.config.rollout_episodes
        )

        train_metrics = ERLTrainMetric(
            pop_episode_lengths=ec_eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=ec_eval_metrics.episode_returns.mean(-1),
        )

        # ======== RL update ========
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

        # ======== EC update ========
        # inject RL into EC

        ec_fitnesses = ec_eval_metrics.episode_returns.mean(axis=-1)
        rl_fitnesses = rl_eval_metrics.episode_returns.mean(axis=-1)
        fitnesses, ec_opt_state = self._rl_injection(
            ec_opt_state, agent_state, ec_fitnesses, rl_fitnesses
        )

        ec_metrics, ec_opt_state = self.ec_optimizer.tell_external(
            ec_opt_state, fitnesses
        )

        train_metrics = train_metrics.replace(
            ec_info=ec_metrics, rb_size=replay_buffer_state.buffer_size
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
        key, rl_eval_key, ec_eval_key = jax.random.split(state.key, num=3)

        rl_eval_metrics = jax.vmap(
            self.evaluator.evaluate, in_axes=(self.agent_state_vmap_axes, 0, None)
        )(
            state.agent_state,
            jax.random.split(rl_eval_key, num=self.config.num_rl_agents),
            self.config.eval_episodes,
        )

        pop_mean_actor_params = state.ec_opt_state.mean

        pop_mean_agent_state = erl_replace_td3_actor_params(
            state.agent_state, pop_mean_actor_params
        )

        ec_eval_metrics = self.evaluator.evaluate(
            pop_mean_agent_state, ec_eval_key, num_episodes=self.config.eval_episodes
        )

        eval_metrics = EvaluateMetric(
            rl_episode_returns=rl_eval_metrics.episode_returns.mean(-1),
            rl_episode_lengths=rl_eval_metrics.episode_lengths.mean(-1),
            pop_center_episode_returns=ec_eval_metrics.episode_returns.mean(),
            pop_center_episode_lengths=ec_eval_metrics.episode_lengths.mean(),
        )

        state = state.replace(key=key)

        return eval_metrics, state

    def learn(self, state: State) -> State:
        sampled_episodes_per_iter = (
            self.config.episodes_for_fitness * self.config.pop_size
            + self.config.rollout_episodes * self.config.num_rl_agents
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
