import logging
import math

from flax import nnx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.distributed.gradients import pmean_nnx_loss_and_grad
from evorl.distributed import psum
from evorl.distributed.sharding import mark_varying
from evorl.metrics import MetricBase
from evorl.rollout import rollout
from evorl.types import (
    PyTreeDict,
    State,
)
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient, rng_fold_in_axis
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update
from evorl.recorders import add_prefix

from evorl.algorithms.offpolicy_utils import clean_trajectory, skip_replay_buffer_state
from evorl.algorithms.td3 import (
    TD3TrainMetric,
    TD3Workflow,
)

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
        rollout_key, rb_key, critic_key, actor_key = rng_fold_in_axis(
            (rollout_key, rb_key, critic_key, actor_key), self.dp_axis_name
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
                    axis_name=self.dp_axis_name,
                )
            )

        replay_buffer_state = self.replay_buffer.add(
            state.replay_buffer_state, trajectory
        )

        def _update_critic_fn(agent_state, opt_state, sample_batch, key):
            critic_opt_state = opt_state.critic

            models = nnx.merge(agent_state.model_state, copy=True)

            def loss_fn(models, sample_batch, key):
                losses = self.agent.critic_loss(
                    agent_state.replace(model_state=nnx.unpack(models)),
                    sample_batch,
                    key,
                )
                return losses.critic_loss, losses

            value, grads = pmean_nnx_loss_and_grad(
                loss_fn,
                self.dp_axis_name,
                has_aux=True,
                wrt=nnx.All(nnx.Param, nnx.PathContains("critic")),
            )(models, sample_batch, key)
            optimizer = nnx.merge(critic_opt_state, copy=True)
            optimizer.update(models.critic, grads["critic"])
            agent_state = agent_state.replace(model_state=nnx.unpack(models))
            critic_loss, critic_loss_dict = value
            critic_opt_state = nnx.unpack(optimizer)

            opt_state = opt_state.replace(critic=critic_opt_state)

            return (
                critic_loss,
                critic_loss_dict,
                agent_state,
                opt_state,
            )

        def _update_actor_fn(agent_state, opt_state, sample_batch, key):
            actor_opt_state = opt_state.actor

            models = nnx.merge(agent_state.model_state, copy=True)
            critic_state = nnx.clone(nnx.state(models.critic, nnx.Not(nnx.Param)))

            def loss_fn(models, sample_batch, key):
                losses = self.agent.actor_loss(
                    agent_state.replace(model_state=nnx.unpack(models)),
                    sample_batch,
                    key,
                )
                return losses.actor_loss, losses

            value, grads = pmean_nnx_loss_and_grad(
                loss_fn,
                self.dp_axis_name,
                has_aux=True,
                wrt=nnx.All(nnx.Param, nnx.PathContains("actor")),
            )(models, sample_batch, key)
            optimizer = nnx.merge(actor_opt_state, copy=True)
            optimizer.update(models.actor, grads["actor"])
            # The actor update only retains the actor's auxiliary variables.
            nnx.update(models.critic, critic_state)
            actor_loss, actor_loss_dict = value
            actor_opt_state = nnx.unpack(optimizer)

            for target, online in (
                (models.target_actor, models.actor),
                (models.target_critic, models.critic),
            ):
                nnx.update(
                    target,
                    soft_target_update(
                        nnx.state(target, nnx.Param),
                        nnx.state(online, nnx.Param),
                        self.config.tau,
                    ),
                )
            agent_state = agent_state.replace(model_state=nnx.unpack(models))

            opt_state = opt_state.replace(actor=actor_opt_state)

            return (
                actor_loss,
                actor_loss_dict,
                agent_state,
                opt_state,
            )

        def _dummy_update_actor_fn(agent_state, opt_state, sample_batch, key):
            actor_loss = jnp.full((), fill_value=MISSING_LOSS)
            local_actor_loss = mark_varying(
                actor_loss, jax.sharding.get_abstract_mesh().manual_axes
            )
            actor_loss_dict = PyTreeDict(actor_loss=local_actor_loss)
            if not self.is_multi_device:
                actor_loss = local_actor_loss

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
        ).all_reduce(axis_name=self.dp_axis_name)

        # calculate the number of timestep
        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.dp_axis_name,
        )

        # iterations is the number of updates of the agent
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(axis_name=self.dp_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            replay_buffer_state=replay_buffer_state,
            opt_state=opt_state,
        )

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
