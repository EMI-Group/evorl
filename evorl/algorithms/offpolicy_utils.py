import logging
import math
from omegaconf import DictConfig

import jax
import jax.numpy as jnp

from evorl.replay_buffers import ReplayBufferState
from evorl.envs import Discrete
from evorl.distributed.comm import psum
from evorl.workflows import OffPolicyWorkflow
from evorl.sample_batch import SampleBatch
from evorl.types import State, PyTreeDict
from evorl.rollout import rollout
from evorl.utils.rl_toolkits import flatten_rollout_trajectory
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient, scan_and_last
from evorl.agent import RandomAgent
from evorl.recorders import add_prefix


logger = logging.getLogger(__name__)


class OffPolicyWorkflowTemplate(OffPolicyWorkflow):
    """Wrapping some common template for off-policy RL with TD Learning."""

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        """Convert single-device totals to per-device rollout and buffer sizes."""
        super()._rescale_config(config)
        num_devices = jax.local_device_count()

        for name in ("num_envs", "replay_buffer_capacity", "batch_size"):
            value = config[name]
            if value < num_devices:
                raise ValueError(
                    f"{name}({value}) must be at least num_devices({num_devices})"
                )
        for name in ("random_timesteps", "learning_start_timesteps"):
            value = config[name]
            if 0 < value < num_devices:
                raise ValueError(
                    f"{name}({value}) must be at least num_devices({num_devices}) when positive"
                )
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

        if config.num_envs % num_devices != 0:
            logger.warning(
                f"num_envs({config.num_envs}) cannot be divided by num_devices({num_devices}), "
                f"rescale num_envs to {config.num_envs // num_devices}"
            )
        if config.replay_buffer_capacity % num_devices != 0:
            logger.warning(
                f"replay_buffer_capacity({config.replay_buffer_capacity}) cannot be divided by num_devices({num_devices}), "
                f"rescale replay_buffer_capacity to {config.replay_buffer_capacity // num_devices}"
            )
        if config.batch_size % num_devices != 0:
            logger.warning(
                f"batch_size({config.batch_size}) cannot be divided by num_devices({num_devices}), "
                f"rescale batch_size to {config.batch_size // num_devices}"
            )
        if config.random_timesteps % num_devices != 0:
            logger.warning(
                f"random_timesteps({config.random_timesteps}) cannot be divided by num_devices({num_devices}), "
                f"rescale random_timesteps to {config.random_timesteps // num_devices}"
            )
        if config.learning_start_timesteps % num_devices != 0:
            logger.warning(
                f"learning_start_timesteps({config.learning_start_timesteps}) cannot be divided by num_devices({num_devices}), "
                f"rescale learning_start_timesteps to {config.learning_start_timesteps // num_devices}"
            )

        config.num_envs = config.num_envs // num_devices
        config.replay_buffer_capacity = config.replay_buffer_capacity // num_devices
        config.batch_size = config.batch_size // num_devices
        config.random_timesteps = config.random_timesteps // num_devices
        config.learning_start_timesteps = config.learning_start_timesteps // num_devices

    def _setup_replaybuffer(self, key: jax.Array) -> ReplayBufferState:
        action_space = self.env.action_space
        obs_space = self.env.obs_space

        # create dummy data to initialize the replay buffer
        if isinstance(action_space, Discrete):
            dummy_action = jnp.zeros((), dtype=jnp.int32)
        else:
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

        def _rollout(agent, agent_state, key, rollout_length):
            env_key, rollout_key = jax.random.split(key)

            env_state = self.env.reset(env_key)

            trajectory, env_state = rollout(
                env_fn=self.env.step,
                action_fn=agent.compute_actions,
                env_state=env_state,
                agent_state=agent_state,
                key=rollout_key,
                rollout_length=rollout_length,
                env_extra_fields=("ori_obs", "termination"),
            )

            # [T, B, ...] -> [T*B, ...]
            trajectory = clean_trajectory(trajectory)
            trajectory = flatten_rollout_trajectory(trajectory)
            trajectory = tree_stop_gradient(trajectory)

            return trajectory

        def _update_obs_preprocessor(agent_state, trajectory):
            if (
                agent_state.obs_preprocessor_state is not None
                and len(trajectory.obs) > 0
            ):
                agent_state = agent_state.replace(
                    obs_preprocessor_state=running_statistics.update(
                        agent_state.obs_preprocessor_state,
                        trajectory.obs,
                        axis_name=self.dp_axis_name,
                    )
                )
            return agent_state

        # ==== fill random transitions ====

        key, random_rollout_key, rollout_key = jax.random.split(state.key, num=3)
        if self.is_multi_device:
            rank = jax.lax.axis_index(self.dp_axis_name)
            random_rollout_key = jax.random.fold_in(random_rollout_key, rank)
            rollout_key = jax.random.fold_in(rollout_key, rank)
        rollout_length = config.random_timesteps // config.num_envs
        if rollout_length > 0:
            random_agent = RandomAgent()
            random_agent_state = random_agent.init(
                obs_space, action_space, jax.random.key(0)
            )

            trajectory = _rollout(
                random_agent,
                random_agent_state,
                key=random_rollout_key,
                rollout_length=rollout_length,
            )

            agent_state = _update_obs_preprocessor(agent_state, trajectory)
            replay_buffer_state = self.replay_buffer.add(
                replay_buffer_state, trajectory
            )

        rollout_timesteps = rollout_length * config.num_envs
        sampled_timesteps = psum(
            jnp.uint32(rollout_timesteps), axis_name=self.dp_axis_name
        )

        # ==== fill tansition state from init agent ====
        rollout_length = max(
            0,
            math.ceil(
                (config.learning_start_timesteps - rollout_timesteps) / config.num_envs
            ),
        )
        if rollout_length > 0:
            trajectory = _rollout(
                self.agent,
                agent_state,
                key=rollout_key,
                rollout_length=rollout_length,
            )

            agent_state = _update_obs_preprocessor(agent_state, trajectory)
            replay_buffer_state = self.replay_buffer.add(
                replay_buffer_state, trajectory
            )

        rollout_timesteps = rollout_length * config.num_envs
        sampled_timesteps = sampled_timesteps + psum(
            jnp.uint32(rollout_timesteps), axis_name=self.dp_axis_name
        )

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
        ).all_reduce(axis_name=self.dp_axis_name)

        return state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
        )

    def _multi_steps(self, state):
        def _step(state, _):
            train_metrics, state = self.step(state)
            return state, train_metrics

        state, train_metrics = scan_and_last(
            _step, state, (), length=self.config.fold_iters
        )

        # jax.debug.print("train_metrics: {}", tree_has_nan(train_metrics))
        # jax.debug.print("state: {}", tree_has_nan(state))

        return train_metrics, state

    def learn(self, state: State) -> State:
        num_devices = self.num_devices
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        sampled_timesteps = state.metrics.sampled_timesteps.tolist()
        num_fold_iters = math.ceil(
            (self.config.total_timesteps - sampled_timesteps)
            / (one_step_timesteps * self.config.fold_iters * num_devices)
        )
        start_iteration = state.metrics.iterations.tolist()
        final_iteration = start_iteration + num_fold_iters * self.config.fold_iters

        for i in range(num_fold_iters):
            train_metrics, state = self._multi_steps(state)
            workflow_metrics = state.metrics

            # current iteration
            iterations = state.metrics.iterations.tolist()
            self.recorder.write(train_metrics.to_local_dict(), iterations)
            self.recorder.write(workflow_metrics.to_local_dict(), iterations)

            if (
                iterations % self.config.eval_interval == 0
                or iterations == final_iteration
            ):
                eval_metrics, state = self.evaluate(state)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iterations
                )

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)
            self.checkpoint_manager.save(
                iterations, saved_state, force=iterations == final_iteration
            )

        return state

    @classmethod
    def enable_jit(cls) -> None:
        super().enable_jit()
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )
        cls._multi_steps = jax.jit(cls._multi_steps, static_argnums=(0,))


def skip_replay_buffer_state(state: State) -> State:
    """Utility function to remove replay_buffer_state from state.

    Usually used when saving the off-policy workflow state to disk.
    """
    return state.replace(replay_buffer_state=None)


def clean_trajectory(trajectory: SampleBatch) -> SampleBatch:
    """Clean the trajectory to make it suitable for the replay buffer."""
    return trajectory.replace(
        next_obs=None,
        dones=None,
    )
