import math
import numpy as np
from typing_extensions import Self  # pytype: disable=not-supported-yet]
from omegaconf import DictConfig

import flashbax
import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp

from evorl.distributed import agent_gradient_update
from evorl.ec.optimizers.utils import ExponetialScheduleSpec
from evorl.metrics import MetricBase
from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import tree_get, tree_set, rng_split_like_tree
from evorl.utils.rl_toolkits import soft_target_update
from evorl.utils.flashbax_utils import get_buffer_size
from evorl.evaluator import Evaluator
from evorl.agent import Agent, AgentState
from evorl.envs import create_env, AutoresetMode, Box
from evorl.recorders import get_1d_array_statistics, add_prefix
from evorl.ec.optimizers import DiagCEM

from ..offpolicy_utils import skip_replay_buffer_state
from ..td3 import TD3Agent, TD3NetworkParams
from .episode_collector import EpisodeCollector
from .cemrl_base import CEMRLWorkflowBase, POPTrainMetric


class EvaluateMetric(MetricBase):
    pop_center_episode_returns: chex.Array
    pop_center_episode_lengths: chex.Array


class CEMRLWorkflow(CEMRLWorkflowBase):
    """
    1 critic + n actors + 1 replay buffer.
    We use shard_map to split and parallel the population.
    """

    @classmethod
    def name(cls):
        return "CEM-RL"

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        """
        return workflow
        """

        # env for one actor
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

        ec_optimizer = DiagCEM(
            pop_size=config.pop_size,
            num_elites=config.num_elites,
            diagonal_variance=ExponetialScheduleSpec(**config.diagonal_variance),
            weighted_update=config.weighted_update,
            rank_weight_shift=config.rank_weight_shift,
            mirror_sampling=config.mirror_sampling,
        )

        if config.fitness_with_exploration:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        ec_evaluator = EpisodeCollector(
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
            ec_evaluator,
            evaluator,
            replay_buffer,
            config,
        )

        workflow.agent_state_pytree_axes = AgentState(
            params=TD3NetworkParams(
                critic_params=None,
                actor_params=0,
                target_critic_params=None,
                target_actor_params=0,
            ),
            obs_preprocessor_state=None,
        )

        workflow._rl_update_fn = build_rl_update_fn(
            agent, optimizer, config, workflow.agent_state_pytree_axes
        )

        return workflow

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree]:
        agent_key, ec_key = jax.random.split(key, 2)

        # one actor + one critic
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )

        init_actor_params = agent_state.params.actor_params
        ec_opt_state = self.ec_optimizer.init(init_actor_params)

        # replace
        pop_actor_params = self.ec_optimizer.ask(ec_opt_state, ec_key)

        agent_state = replace_actor_params(agent_state, pop_actor_params)

        opt_state = PyTreeDict(
            # Note: we create and drop the actors' opt_state at every step
            critic=self.optimizer.init(agent_state.params.critic_params),
        )

        return agent_state, opt_state, ec_opt_state

    def _ec_sample(self, ec_opt_state, key):
        pop_actor_params = self.ec_optimizer.ask(ec_opt_state, key)

        # Note: Avoid always choosing the positve parts for learning
        if self.config.mirror_sampling:
            keys = rng_split_like_tree(key, pop_actor_params)
            pop_actor_params = jtu.tree_map(
                lambda x, k: jax.random.permutation(k, x, axis=0),
                pop_actor_params,
                keys,
            )

        return pop_actor_params

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

        pop_actor_params = agent_state.params.actor_params

        key, rollout_key, cem_key, learn_key = jax.random.split(state.key, num=4)

        # ======== RL update ========
        if iterations > self.config.warmup_iters:
            learning_actor_slice = slice(
                self.config.num_learning_offspring
            )  # [:self.config.num_learning_offspring]
            learning_actor_params = tree_get(pop_actor_params, learning_actor_slice)
            learning_agent_state = replace_actor_params(
                agent_state, learning_actor_params
            )

            # reset and add actors' opt_state
            new_opt_state = opt_state.replace(
                actor=self.optimizer.init(learning_actor_params),
            )

            td3_metrics, learning_agent_state, new_opt_state = self._rl_update(
                learning_agent_state, new_opt_state, replay_buffer_state, learn_key
            )

            pop_actor_params = tree_set(
                pop_actor_params,
                learning_agent_state.params.actor_params,
                learning_actor_slice,
                unique_indices=True,
            )
            # Note: updated critic_params are stored in learning_agent_state
            # actor_params [num_learning_offspring, ...] -> [pop_size, ...]
            # reset target_actor_params
            agent_state = replace_actor_params(learning_agent_state, pop_actor_params)

            # drop the actors' opt_state
            opt_state = opt_state.replace(
                critic=new_opt_state.critic,
            )

        else:
            td3_metrics = None

        # ======== CEM update ========
        # the trajectory [T, #pop*B, ...]
        # metrics: [#pop, B]
        eval_metrics, trajectory = self._rollout(agent_state, rollout_key)

        fitnesses = eval_metrics.episode_returns.mean(axis=-1)

        replay_buffer_state = self._add_to_replay_buffer(
            replay_buffer_state,
            trajectory,
            eval_metrics.episode_lengths.flatten(),
        )

        train_metrics = POPTrainMetric(
            rb_size=get_buffer_size(replay_buffer_state),
            pop_episode_lengths=eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=eval_metrics.episode_returns.mean(-1),
            rl_metrics=td3_metrics,
        )

        ec_opt_state = self._ec_update(ec_opt_state, pop_actor_params, fitnesses)

        new_pop_actor_params = self._ec_sample(ec_opt_state, cem_key)
        agent_state = replace_actor_params(agent_state, new_pop_actor_params)

        # adding debug info for CEM
        ec_info = PyTreeDict()
        ec_info.cov_noise = ec_opt_state.cov_noise
        if td3_metrics is not None:
            elites_indices = jax.lax.top_k(fitnesses, self.config.num_elites)[1]
            elites_from_rl = jnp.isin(
                jnp.arange(self.config.num_learning_offspring), elites_indices
            )
            ec_info = ec_info.replace(
                elites_from_rl=elites_from_rl.sum(),
                elites_from_rl_ratio=elites_from_rl.mean(),
            )

        train_metrics = train_metrics.replace(ec_info=ec_info)

        # calculate the number of timestep
        sampled_timesteps = eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        sampled_episodes = jnp.uint32(self.config.episodes_for_fitness * pop_size)

        # iterations is the number of updates of the agent

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
        pop_mean_actor_params = state.ec_opt_state.mean

        pop_mean_agent_state = replace_actor_params(
            state.agent_state, pop_mean_actor_params
        )

        key, eval_key = jax.random.split(state.key, num=2)

        # [#episodes]
        raw_eval_metrics = self.evaluator.evaluate(
            pop_mean_agent_state, num_episodes=self.config.eval_episodes, key=eval_key
        )

        eval_metrics = EvaluateMetric(
            pop_center_episode_returns=raw_eval_metrics.episode_returns.mean(),
            pop_center_episode_lengths=raw_eval_metrics.episode_lengths.mean(),
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

            if train_metrics_dict["rl_metrics"] is not None:
                train_metrics_dict["rl_metrics"]["actor_loss"] /= (
                    self.config.num_learning_offspring
                )
                train_metrics_dict["rl_metrics"]["raw_loss_dict"] = jtu.tree_map(
                    get_1d_array_statistics,
                    train_metrics_dict["rl_metrics"]["raw_loss_dict"],
                )

            self.recorder.write(train_metrics_dict, iters)

            std_statistics = get_std_statistics(state.ec_opt_state.variance["params"])
            self.recorder.write({"ec/std": std_statistics}, iters)

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
        params=agent_state.params.replace(
            actor_params=pop_actor_params,
            target_actor_params=pop_actor_params,
        )
    )


def build_rl_update_fn(
    agent: Agent,
    optimizer: optax.GradientTransformation,
    config: DictConfig,
    agent_state_pytree_axes: AgentState,
):
    num_learning_offspring = config.num_learning_offspring

    def critic_loss_fn(agent_state, sample_batch, key):
        # loss on a single critic with multiple actors
        # sample_batch: (B, ...)

        loss_dict = jax.vmap(
            agent.critic_loss, in_axes=(agent_state_pytree_axes, None, 0)
        )(agent_state, sample_batch, jax.random.split(key, num_learning_offspring))

        # mean over the num_learning_offspring
        loss = config.loss_weights.critic_loss * loss_dict.critic_loss.mean()

        return loss, loss_dict

    def actor_loss_fn(agent_state, sample_batch, key):
        # loss on a single actor
        # different actor shares same sample_batch (B, ...) input
        loss_dict = jax.vmap(
            agent.actor_loss, in_axes=(agent_state_pytree_axes, None, 0)
        )(agent_state, sample_batch, jax.random.split(key, num_learning_offspring))

        # sum over the num_learning_offspring
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


def get_std_statistics(variance):
    def _get_stats(x):
        x = np.sqrt(x)
        return dict(
            min=np.min(x).tolist(),
            max=np.max(x).tolist(),
            mean=np.mean(x).tolist(),
        )

    return jtu.tree_map(_get_stats, variance)
