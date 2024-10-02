import copy
import logging
import math
from typing import Any
from typing_extensions import Self  # pytype: disable=not-supported-yet]
from omegaconf import DictConfig

import chex
import flashbax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp

from evorl.distributed import agent_gradient_update
from evorl.metrics import MetricBase, metricfield
from evorl.types import (
    PyTreeDict,
    State,
)
from evorl.utils import running_statistics
from evorl.utils.jax_utils import (
    scan_and_mean,
    tree_stop_gradient,
    tree_get,
    tree_set,
)
from evorl.utils.rl_toolkits import (
    soft_target_update,
    flatten_rollout_trajectory,
)
from evorl.utils.flashbax_utils import get_buffer_size
from evorl.evaluator import Evaluator
from evorl.sample_batch import SampleBatch
from evorl.agent import Agent, AgentState, RandomAgent
from evorl.envs import create_env, AutoresetMode, Box, Env
from evorl.workflows import Workflow
from evorl.rollout import rollout
from evorl.recorders import get_1d_array_statistics, add_prefix
from evorl.ec.optimizers.cem import DiagCEM, EvoOptimizer

from ..td3 import TD3TrainMetric, TD3Agent, TD3NetworkParams
from ..offpolicy_utils import clean_trajectory, skip_replay_buffer_state
from .episode_collector import EpisodeCollector
from .utils import flatten_pop_rollout_episode, get_std_statistics


logger = logging.getLogger(__name__)


class POPTrainMetric(MetricBase):
    rb_size: int
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array
    rl_metrics: MetricBase | None = None
    ec_info: PyTreeDict = metricfield(default_factory=PyTreeDict)


class WorkflowMetric(MetricBase):
    sampled_timesteps: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class EvaluateMetric(MetricBase):
    pop_center_episode_returns: chex.Array
    pop_center_episode_lengths: chex.Array


class CEMRLWorkflow(Workflow):
    """
    1 critic + n actors + 1 replay buffer.
    We use shard_map to split and parallel the population.
    """

    def __init__(
        self,
        env: Env,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        ec_optimizer: EvoOptimizer,
        collector: EpisodeCollector,
        evaluator: Evaluator,  # to evaluate the pop-mean actor
        replay_buffer: Any,
        config: DictConfig,
    ):
        super().__init__(config)
        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.ec_optimizer = ec_optimizer
        self.collector = collector
        self.evaluator = evaluator
        self.replay_buffer = replay_buffer

        self.devices = jax.local_devices()[:1]

    @classmethod
    def name(cls):
        return "CEM-RL"

    @classmethod
    def build_from_config(
        cls,
        config: DictConfig,
        enable_multi_devices: bool = False,
        enable_jit: bool = True,
    ) -> Self:
        config = copy.deepcopy(config)  # avoid in-place modification

        devices = jax.local_devices()

        if enable_multi_devices or len(devices) > 1:
            raise NotImplementedError("Multi-devices is not supported yet.")

        if enable_jit:
            cls.enable_jit()

        workflow = cls._build_from_config(config)

        return workflow

    @classmethod
    def _build_from_config(cls, config: DictConfig):
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
            init_diagonal_variance=config.diagonal_variance.init,
            final_diagonal_variance=config.diagonal_variance.final,
            diagonal_variance_decay=config.diagonal_variance.decay,
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

        workflow._rl_update_fn = _build_rl_update_fn(
            agent, optimizer, config, workflow.agent_state_pytree_axes
        )

        return workflow

    def setup(self, key: chex.PRNGKey) -> State:
        """
        obs_preprocessor_state update strategy: only updated at _postsetup_replaybuffer(), then fixed during the training.
        """
        key, agent_key, rb_key = jax.random.split(key, 3)

        # [#pop, ...]
        agent_state, opt_state, ec_opt_state = self._setup_agent_and_optimizer(
            agent_key
        )

        workflow_metrics = WorkflowMetric()

        replay_buffer_state = self._setup_replaybuffer(rb_key)

        # =======================

        state = State(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            opt_state=opt_state,
            ec_opt_state=ec_opt_state,
            replay_buffer_state=replay_buffer_state,
        )

        if self.config.random_timesteps > 0:
            logger.info("Start replay buffer post-setup")
            state = self._postsetup_replaybuffer(state)
            logger.info("Complete replay buffer post-setup")

        return state

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

    def _setup_replaybuffer(self, key: chex.PRNGKey) -> chex.ArrayTree:
        action_space = self.env.action_space
        obs_space = self.env.obs_space

        # create dummy data to initialize the replay buffer
        dummy_action = jnp.zeros(action_space.shape)
        dummy_obs = jnp.zeros(obs_space.shape)

        dummy_reward = jnp.zeros(())
        dummy_done = jnp.zeros(())

        dummy_sample_batch = SampleBatch(
            obs=dummy_obs,
            actions=dummy_action,
            rewards=dummy_reward,
            # next_obs=dummy_obs,
            # dones=dummy_done,
            extras=PyTreeDict(
                policy_extras=PyTreeDict(),
                env_extras=PyTreeDict(
                    {"ori_obs": dummy_obs, "termination": dummy_done}
                ),
            ),
        )
        replay_buffer_state = self.replay_buffer.init(dummy_sample_batch)

        return replay_buffer_state

    def _postsetup_replaybuffer(self, state: State) -> State:
        action_space = self.env.action_space
        obs_space = self.env.obs_space
        config = self.config

        replay_buffer_state = state.replay_buffer_state
        agent_state = state.agent_state

        # We need a separate autoreset env to fill the replay buffer
        env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )

        # ==== fill random transitions ====

        key, env_key, rollout_key = jax.random.split(state.key, num=3)
        random_agent = RandomAgent()
        random_agent_state = random_agent.init(
            obs_space, action_space, jax.random.PRNGKey(0)
        )
        rollout_length = config.random_timesteps // config.num_envs

        env_state = env.reset(env_key)
        trajectory, env_state = rollout(
            env_fn=env.step,
            action_fn=random_agent.compute_actions,
            env_state=env_state,
            agent_state=random_agent_state,
            key=rollout_key,
            rollout_length=rollout_length,
            env_extra_fields=("ori_obs", "termination"),
        )

        # [T, B, ...] -> [T*B, ...]
        trajectory = clean_trajectory(trajectory)
        trajectory = flatten_rollout_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        replay_buffer_state = self.replay_buffer.add(replay_buffer_state, trajectory)

        if agent_state.obs_preprocessor_state is not None and rollout_length > 0:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                )
            )

        sampled_timesteps = jnp.uint32(rollout_length * config.num_envs)
        # Since we sample from autoreset env, this metric might not be accurate:
        sampled_episodes = trajectory.dones.astype(jnp.uint32).sum()

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
        )

        return state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
        )

    def _rollout(self, agent_state, key):
        eval_metrics, trajectory = jax.vmap(
            self.collector.evaluate,
            in_axes=(self.agent_state_pytree_axes, None, 0),
        )(
            agent_state,
            self.config.episodes_for_fitness,
            jax.random.split(key, self.config.pop_size),
        )

        trajectory = clean_trajectory(trajectory)
        # [#pop, T, B, ...] -> [T, #pop*B, ...]
        trajectory = flatten_pop_rollout_episode(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        return eval_metrics, trajectory

    def _rl_update(self, agent_state, opt_state, replay_buffer_state, key):
        """
        sample_batches: (num_rl_updates_per_iter, actor_update_interval, B, ...)
        """

        def _sample_fn(key):
            return self.replay_buffer.sample(replay_buffer_state, key).experience

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            key, rb_key, learn_key = jax.random.split(key, 3)

            rb_key = jax.random.split(key, self.config.actor_update_interval)
            sample_batches = jax.vmap(_sample_fn)(rb_key)

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

    def _ec_update(self, ec_opt_state, pop_actor_params, fitnesses):
        return self.ec_optimizer.tell(ec_opt_state, pop_actor_params, fitnesses)

    def _ec_sample(self, ec_opt_state, key):
        return self.ec_optimizer.ask(ec_opt_state, key)

    def _add_to_replay_buffer(self, replay_buffer_state, trajectory, episode_lengths):
        # trajectory [T,B,...]
        # episode_lengths [B]

        def concat_valid(x):
            # x: [T, B, ...]
            return jnp.concatenate(
                [x[:t, i] for i, t in enumerate(episode_lengths)], axis=0
            )

        valid_trajectory = jtu.tree_map(concat_valid, trajectory)

        replay_buffer_state = self.replay_buffer.add(
            replay_buffer_state, valid_trajectory
        )

        return replay_buffer_state

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

    @classmethod
    def enable_jit(cls) -> None:
        """
        Do not jit replay buffer add
        """
        cls._rollout = jax.jit(cls._rollout, static_argnums=(0,))
        cls._rl_update = jax.jit(cls._rl_update, static_argnums=(0,))
        cls._ec_sample = jax.jit(cls._ec_sample, static_argnums=(0,))
        cls._ec_update = jax.jit(cls._ec_update, static_argnums=(0,))

        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )


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


def _build_rl_update_fn(
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
