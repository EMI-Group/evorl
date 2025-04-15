import copy
import logging
from typing_extensions import Self  # pytype: disable=not-supported-yet]
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import optax

from evorl.agent import AgentStateAxis
from evorl.metrics import MetricBase, metric_field
from evorl.types import PyTreeDict, State
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory
from evorl.evaluators import Evaluator, EpisodeCollector
from evorl.sample_batch import SampleBatch
from evorl.replay_buffers import AbstractReplayBuffer, ReplayBufferState
from evorl.agent import Agent, AgentState, RandomAgent
from evorl.envs import create_env, AutoresetMode, Env
from evorl.workflows import Workflow
from evorl.rollout import rollout
from evorl.ec.optimizers import EvoOptimizer, ECState

from ..offpolicy_utils import clean_trajectory

logger = logging.getLogger(__name__)


class ERLTrainMetric(MetricBase):
    pop_episode_returns: chex.Array | None = None
    pop_episode_lengths: chex.Array | None = None
    rb_size: chex.Array | None = None
    rl_episode_returns: chex.Array | None = None
    rl_episode_lengths: chex.Array | None = None
    rl_metrics: MetricBase | None = None
    ec_info: PyTreeDict = metric_field(default_factory=PyTreeDict)


class WorkflowMetric(MetricBase):
    sampled_timesteps: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    rl_sampled_timesteps: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class ERLWorkflowBase(Workflow):
    def __init__(
        self,
        *,
        env: Env,
        agent: Agent,
        agent_state_vmap_axes: AgentStateAxis,
        optimizer: optax.GradientTransformation,
        ec_optimizer: EvoOptimizer,
        ec_collector: EpisodeCollector,
        rl_collector: EpisodeCollector,
        evaluator: Evaluator,  # to evaluate the pop-mean actor
        replay_buffer: AbstractReplayBuffer,
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
        key, agent_key, rb_key = jax.random.split(key, 3)

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

    def _setup_replaybuffer(self, key: chex.PRNGKey) -> ReplayBufferState:
        action_space = self.env.action_space
        obs_space = self.env.obs_space

        # create dummy data to initialize the replay buffer
        dummy_action = jnp.zeros(action_space.shape)
        dummy_obs = obs_space.sample(key)

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
            config.env,
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

        # sampled_timesteps = jnp.uint32(rollout_length * config.num_envs)
        # # Since we sample from autoreset env, this metric might not be accurate:
        # sampled_episodes = trajectory.dones.sum()

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

        # Note: we don't count the random transitions in the metrics, to align the x-axis(sampled_episodes/iterations) over different runs
        # workflow_metrics = state.metrics.replace(
        #     sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
        #     sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
        # )

        return state.replace(
            key=key,
            # metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
        )

    def warmup_step(self, state: State) -> tuple[MetricBase, State]:
        raise NotImplementedError

    def _rl_injection(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        raise NotImplementedError

    @classmethod
    def enable_jit(cls) -> None:
        cls.step = jax.jit(cls.step, static_argnums=(0,))

        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )
        cls.warmup_step = jax.jit(cls.warmup_step, static_argnums=(0,))
