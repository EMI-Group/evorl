import copy
import logging
from typing import Any
from typing_extensions import Self  # pytype: disable=not-supported-yet]
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.agent import AgentStateAxis
from evorl.metrics import MetricBase, metricfield
from evorl.types import PyTreeDict, State, Params
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient, scan_and_mean
from evorl.utils.rl_toolkits import flatten_rollout_trajectory
from evorl.utils.ec_utils import flatten_pop_rollout_episode
from evorl.evaluators import Evaluator, EpisodeCollector
from evorl.sample_batch import SampleBatch
from evorl.agent import Agent, AgentState, RandomAgent
from evorl.envs import create_env, AutoresetMode, Env
from evorl.workflows import Workflow
from evorl.rollout import rollout
from evorl.ec.optimizers import EvoOptimizer, ECState

from ..td3 import TD3TrainMetric
from ..offpolicy_utils import clean_trajectory


logger = logging.getLogger(__name__)


class POPTrainMetric(MetricBase):
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array
    rb_size: int = 0
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


class ERLWorkflowBase(Workflow):
    """
    EC: n actors
    RL: k actors + k critics + 1 replay buffer.
    """

    def __init__(
        self,
        env: Env,
        agent: Agent,
        agent_state_vmap_axes: AgentStateAxis,
        optimizer: optax.GradientTransformation,
        ec_optimizer: EvoOptimizer,
        ec_collector: EpisodeCollector,
        rl_collector: EpisodeCollector,
        evaluator: Evaluator,  # to evaluate the pop-mean actor
        replay_buffer: Any,
        config: DictConfig,
    ):
        super().__init__(config)
        self.env = env
        self.agent = agent
        self.agent_state_vmap_axes = agent_state_vmap_axes
        self.optimizer = optimizer
        self.ec_optimizer = ec_optimizer
        self.ec_collector = ec_collector
        self.rl_collector = rl_collector
        self.evaluator = evaluator
        self.replay_buffer = replay_buffer

        self.devices = jax.local_devices()[:1]

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
    def _build_from_config(cls, config: DictConfig) -> Self:
        raise NotImplementedError

    def setup(self, key: chex.PRNGKey) -> State:
        """
        obs_preprocessor_state update strategy: only updated at _postsetup_replaybuffer(), then fixed during the training.
        """
        key, agent_key, rb_key = jax.random.split(key, 3)

        # agent_state: [num_rl_agents, ...]
        # ec_opt_state.pop: [pop_size, ...]
        agent_state, opt_state, ec_opt_state = self._setup_agent_and_optimizer(
            agent_key
        )

        workflow_metrics = self._setup_workflow_metrics()

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

    def _setup_workflow_metrics(self) -> MetricBase:
        return WorkflowMetric()

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree, ECState]:
        raise NotImplementedError

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

        sampled_timesteps = jnp.uint32(rollout_length * config.num_envs)
        # Since we sample from autoreset env, this metric might not be accurate:
        sampled_episodes = trajectory.dones.sum()

        # [T, B, ...] -> [T*B, ...]
        trajectory = clean_trajectory(trajectory)
        trajectory = flatten_rollout_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        replay_buffer_state = self.replay_buffer.add(replay_buffer_state, trajectory)

        if agent_state.obs_preprocessor_state is not None and rollout_length > 0:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state, trajectory.obs
                )
            )

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

    def _ec_rollout(self, agent_state, key):
        eval_metrics, trajectory = jax.vmap(
            self.ec_collector.rollout,
            in_axes=(self.agent_state_vmap_axes, 0, None),
        )(
            agent_state,
            jax.random.split(key, self.config.pop_size),
            self.config.episodes_for_fitness,
        )

        trajectory = clean_trajectory(trajectory)
        # [#pop, T, B, ...] -> [T, #pop*B, ...]
        trajectory = flatten_pop_rollout_episode(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        return eval_metrics, trajectory

    def _rl_rollout(self, agent_state, key):
        eval_metrics, trajectory = jax.vmap(
            self.rl_collector.rollout,
            in_axes=(self.agent_state_vmap_axes, 0, None),
        )(
            agent_state,
            jax.random.split(key, self.config.num_rl_agents),
            self.config.rollout_episodes,
        )

        trajectory = clean_trajectory(trajectory)
        # [num_rl_agents, T, B, ...] -> [T, num_rl_agents*B, ...]
        trajectory = flatten_pop_rollout_episode(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        return eval_metrics, trajectory

    def _rl_update(self, agent_state, opt_state, replay_buffer_state, key):
        def _sample_fn(key):
            return self.replay_buffer.sample(replay_buffer_state, key).experience

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

    def _ec_update(
        self, ec_opt_state: ECState, fitnesses: chex.Array
    ) -> tuple[PyTreeDict, ECState]:
        return self.ec_optimizer.tell(ec_opt_state, fitnesses)

    def _ec_sample(self, ec_opt_state: ECState) -> tuple[Params, ECState]:
        return self.ec_optimizer.ask(ec_opt_state)

    def _rl_injection(self, *args, **kwargs):
        raise NotImplementedError

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

    @classmethod
    def enable_jit(cls) -> None:
        """
        Do not jit replay buffer add
        """
        cls._rl_rollout = jax.jit(cls._rl_rollout, static_argnums=(0,))
        cls._rl_update = jax.jit(cls._rl_update, static_argnums=(0,))
        cls._ec_rollout = jax.jit(cls._ec_rollout, static_argnums=(0,))
        cls._rl_injection = jax.jit(cls._rl_injection, static_argnums=(0,))
        cls._ec_sample = jax.jit(cls._ec_sample, static_argnums=(0,))
        cls._ec_update = jax.jit(cls._ec_update, static_argnums=(0,))

        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )
