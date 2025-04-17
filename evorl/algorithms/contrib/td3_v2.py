import logging
import math

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.distributed import psum, unpmap
from evorl.distributed.gradients import agent_gradient_update
from evorl.metrics import MetricBase
from evorl.rollout import rollout
from evorl.types import (
    PyTreeDict,
    State,
)
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update
from evorl.recorders import add_prefix

from evorl.algorithms.offpolicy_utils import clean_trajectory, skip_replay_buffer_state
from evorl.algorithms.td3 import TD3TrainMetric, TD3Workflow

logger = logging.getLogger(__name__)

MISSING_LOSS = -1e10


class TD3V2Workflow(TD3Workflow):
    """The similar impl of TD3 in SB3 and CleanRL."""

    @classmethod
    def name(cls):
        return "TD3-V2"

    def step(self, state: State) -> tuple[MetricBase, State]:
        iterations = state.metrics.iterations + 1
        key, rollout_key, rb_key, critic_key, actor_key = jax.random.split(
            state.key, num=5
        )

        # the trajectory [T, B, ...]
        trajectory, env_state = rollout(
            env_fn=self.env.step,
            action_fn=self.agent.compute_actions,
            env_state=state.env_state,
            agent_state=state.agent_state,
            key=rollout_key,
            rollout_length=self.config.rollout_length,
            env_extra_fields=("ori_obs", "termination"),
        )

        trajectory = clean_trajectory(trajectory)
        trajectory = flatten_rollout_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        agent_state = state.agent_state
        opt_state = state.opt_state

        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    pmap_axis_name=self.pmap_axis_name,
                )
            )

        replay_buffer_state = self.replay_buffer.add(
            state.replay_buffer_state, trajectory
        )

        def critic_loss_fn(agent_state, sample_batch, key):
            loss_dict = self.agent.critic_loss(agent_state, sample_batch, key)

            loss = loss_dict.critic_loss
            return loss, loss_dict

        def actor_loss_fn(agent_state, sample_batch, key):
            loss_dict = self.agent.actor_loss(agent_state, sample_batch, key)

            loss = loss_dict.actor_loss
            return loss, loss_dict

        critic_update_fn = agent_gradient_update(
            critic_loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True,
            attach_fn=lambda agent_state, critic_params: agent_state.replace(
                params=agent_state.params.replace(critic_params=critic_params)
            ),
            detach_fn=lambda agent_state: agent_state.params.critic_params,
        )

        actor_update_fn = agent_gradient_update(
            actor_loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True,
            attach_fn=lambda agent_state, actor_params: agent_state.replace(
                params=agent_state.params.replace(actor_params=actor_params)
            ),
            detach_fn=lambda agent_state: agent_state.params.actor_params,
        )

        def _update_critic_fn(agent_state, opt_state, sample_batch, key):
            critic_opt_state = opt_state.critic

            (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
                critic_update_fn(critic_opt_state, agent_state, sample_batch, key)
            )

            opt_state = opt_state.replace(critic=critic_opt_state)

            return (
                critic_loss,
                critic_loss_dict,
                agent_state,
                opt_state,
            )

        def _update_actor_fn(agent_state, opt_state, sample_batch, key):
            actor_opt_state = opt_state.actor

            (actor_loss, actor_loss_dict), agent_state, actor_opt_state = (
                actor_update_fn(actor_opt_state, agent_state, sample_batch, key)
            )

            target_actor_params = soft_target_update(
                agent_state.params.target_actor_params,
                agent_state.params.actor_params,
                self.config.tau,
            )
            target_critic_params = soft_target_update(
                agent_state.params.target_critic_params,
                agent_state.params.critic_params,
                self.config.tau,
            )
            agent_state = agent_state.replace(
                params=agent_state.params.replace(
                    target_actor_params=target_actor_params,
                    target_critic_params=target_critic_params,
                )
            )

            opt_state = opt_state.replace(actor=actor_opt_state)

            return (
                actor_loss,
                actor_loss_dict,
                agent_state,
                opt_state,
            )

        def _dummy_update_actor_fn(agent_state, opt_state, sample_batch, key):
            actor_loss = jnp.full((), fill_value=MISSING_LOSS)
            actor_loss_dict = PyTreeDict(actor_loss=actor_loss)

            return (
                actor_loss,
                actor_loss_dict,
                agent_state,
                opt_state,
            )

        sample_batch = self.replay_buffer.sample(replay_buffer_state, rb_key)

        critic_loss, critic_loss_dict, agent_state, opt_state = _update_critic_fn(
            agent_state, opt_state, sample_batch, critic_key
        )

        # Note: using cond prohibits the parallel training by vmap
        (
            actor_loss,
            actor_loss_dict,
            agent_state,
            opt_state,
        ) = jax.lax.cond(
            iterations % self.config.actor_update_interval == 0,
            _update_actor_fn,
            _dummy_update_actor_fn,
            agent_state,
            opt_state,
            sample_batch,
            actor_key,
        )

        train_metrics = TD3TrainMetric(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        # calculate the number of timestep
        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.pmap_axis_name,
        )

        # iterations is the number of updates of the agent
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            replay_buffer_state=replay_buffer_state,
            opt_state=opt_state,
        )

    def learn(self, state: State) -> State:
        num_devices = jax.device_count()
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        sampled_timesteps = unpmap(state.metrics.sampled_timesteps).tolist()
        num_iters = math.ceil(
            (self.config.total_timesteps - sampled_timesteps)
            / (one_step_timesteps * self.config.fold_iters * num_devices)
        )

        start_iteration = unpmap(state.metrics.iterations, self.pmap_axis_name).tolist()
        final_iteration = num_iters + start_iteration

        for i in range(num_iters):
            train_metrics, state = self._multi_steps(state)
            workflow_metrics = state.metrics

            # current iteration
            iterations = unpmap(state.metrics.iterations, self.pmap_axis_name).tolist()
            train_metrics = unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = unpmap(workflow_metrics, self.pmap_axis_name)

            train_metrics = jtu.tree_map(
                lambda x: None if x == MISSING_LOSS else x, train_metrics
            )

            self.recorder.write(train_metrics.to_local_dict(), iterations)
            self.recorder.write(workflow_metrics.to_local_dict(), iterations)

            if (
                iterations % self.config.eval_interval == 0
                or iterations == final_iteration
            ):
                eval_metrics, state = self.evaluate(state)
                eval_metrics = unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iterations
                )

            saved_state = unpmap(state, self.pmap_axis_name)
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)
            self.checkpoint_manager.save(
                iterations, saved_state, force=iterations == final_iteration
            )

        return state
