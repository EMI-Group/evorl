import math
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.replay_buffers import ReplayBuffer
from evorl.metrics import MetricBase
from evorl.types import State, PyTreeDict
from evorl.evaluators import Evaluator, EpisodeCollector
from evorl.agent import AgentState
from evorl.types import Params
from evorl.envs import create_env, AutoresetMode
from evorl.recorders import get_1d_array_statistics, add_prefix
from evorl.ec.optimizers import OpenES, ExponentialScheduleSpec, ECState
from evorl.utils.jax_utils import tree_set, tree_get
from evorl.algorithms.td3 import make_mlp_td3_agent, TD3NetworkParams
from evorl.algorithms.offpolicy_utils import skip_replay_buffer_state

from .cemrl_td3_workflow import (
    create_dummy_td3_trainmetric,
    cemrl_replace_td3_actor_params,
    CEMRLTD3WorkflowTemplate,
)
from ..cemrl_workflow import CEMRLTrainMetric


class CEMRLOpenESWorkflow(CEMRLTD3WorkflowTemplate):
    """1 critic + n actors + 1 replay buffer.

    We use shard_map to split and parallel the population.
    """

    @classmethod
    def name(cls):
        return "CEMRL-OpenES"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        assert config.warmup_iters > 0 or config.random_timesteps > 0, (
            "Either warmup_iters or random_timesteps should be positive to pre-fill some data in the replay buffer"
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
            params=TD3NetworkParams(
                critic_params=None,
                actor_params=0,
                target_critic_params=None,
                target_actor_params=0,
            ),
            obs_preprocessor_state=None,
        )

        workflow = cls(
            env=env,
            agent=agent,
            agent_state_vmap_axes=agent_state_vmap_axes,
            optimizer=optimizer,
            ec_optimizer=ec_optimizer,
            collector=collector,
            evaluator=evaluator,
            replay_buffer=replay_buffer,
            config=config,
        )

        return workflow

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree, ECState]:
        agent_key, ec_key = jax.random.split(key)

        # one actor + one critic
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )

        init_actor_params = agent_state.params.actor_params
        ec_opt_state = self.ec_optimizer.init(init_actor_params, ec_key)

        agent_state = cemrl_replace_td3_actor_params(agent_state, pop_actor_params=None)

        opt_state = PyTreeDict(
            # Note: we create and drop the actors' opt_state at every step
            critic=self.optimizer.init(agent_state.params.critic_params),
            actor=None,
        )

        return agent_state, opt_state, ec_opt_state

    def _rl_injection(
        self, ec_opt_state: ECState, pop: Params, external_indices
    ) -> ECState:
        external_noise = jtu.tree_map(
            lambda x, m: (x - m) / ec_opt_state.noise_std,
            tree_get(pop, external_indices),
            ec_opt_state.mean,
        )
        noise = tree_set(
            ec_opt_state.noise,
            external_noise,
            external_indices,
            unique_indices=True,
        )

        return ec_opt_state.replace(noise=noise)

    def step(self, state: State) -> tuple[MetricBase, State]:
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        opt_state = state.opt_state
        ec_opt_state = state.ec_opt_state
        replay_buffer_state = state.replay_buffer_state
        iterations = state.metrics.iterations + 1

        pop_actor_params = agent_state.params.actor_params

        key, rollout_key, perm_key, learn_key = jax.random.split(state.key, num=4)

        # ======= CEM Sample ========
        pop_actor_params, ec_opt_state = self.ec_optimizer.ask(ec_opt_state)

        # ======== RL update ========
        learning_actor_indices = jax.random.choice(
            perm_key,
            self.config.pop_size,
            (self.config.num_learning_offspring,),
            replace=False,
        )

        def _rl_update(agent_state, opt_state, pop_actor_params):
            learning_actor_params = tree_get(pop_actor_params, learning_actor_indices)
            learning_agent_state = cemrl_replace_td3_actor_params(
                agent_state, learning_actor_params
            )

            # reset actors' opt_state
            learning_opt_state = opt_state.replace(
                actor=self.optimizer.init(learning_actor_params),
            )

            td3_metrics, learning_agent_state, learning_opt_state = self._rl_update(
                learning_agent_state,
                learning_opt_state,
                replay_buffer_state,
                learn_key,
            )

            pop_actor_params = tree_set(
                pop_actor_params,
                learning_agent_state.params.actor_params,
                learning_actor_indices,
                unique_indices=True,
            )
            # drop the actors and their opt_state
            agent_state = cemrl_replace_td3_actor_params(
                learning_agent_state, pop_actor_params=None
            )
            opt_state = learning_opt_state.replace(actor=None)
            return td3_metrics, pop_actor_params, agent_state, opt_state

        def _dummy_rl_update(agent_state, opt_state, pop_actor_params):
            return (
                create_dummy_td3_trainmetric(self.config.num_learning_offspring),
                pop_actor_params,
                agent_state,
                opt_state,
            )

        td3_metrics, pop_actor_params, agent_state, opt_state = jax.lax.cond(
            iterations > self.config.warmup_iters,
            _rl_update,
            _dummy_rl_update,
            agent_state,
            opt_state,
            pop_actor_params,
        )

        # ======== CEM update ========
        pop_agent_state = cemrl_replace_td3_actor_params(agent_state, pop_actor_params)

        # the trajectory [T, #pop*B, ...]
        # metrics: [#pop, B]
        eval_metrics, trajectory, replay_buffer_state = self._rollout(
            pop_agent_state, replay_buffer_state, rollout_key
        )

        fitnesses = eval_metrics.episode_returns.mean(axis=-1)

        ec_opt_state = jax.lax.cond(
            iterations > self.config.warmup_iters,
            self._rl_injection,
            lambda ec_opt_state, pop, external_indices: ec_opt_state,
            ec_opt_state,
            pop_actor_params,
            learning_actor_indices,
        )
        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        train_metrics = CEMRLTrainMetric(
            rb_size=replay_buffer_state.buffer_size,
            pop_episode_lengths=eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=eval_metrics.episode_returns.mean(-1),
            rl_metrics=td3_metrics,
            ec_info=ec_metrics,
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
