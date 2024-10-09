import logging
from functools import partial
import math

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import orbax.checkpoint as ocp

from evorl.distributed import (
    agent_gradient_update,
    psum,
    tree_unpmap,
)
from evorl.agent import AgentState, RandomAgent
from evorl.types import PyTreeDict, State
from evorl.metrics import MetricBase, EvaluateMetric
from evorl.rollout import rollout
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient, scan_and_mean
from evorl.utils.rl_toolkits import (
    soft_target_update,
    flatten_rollout_trajectory,
    flatten_pop_rollout_trajectory,
)
from evorl.recorders import add_prefix, get_1d_array_statistics

from ..offpolicy_utils import clean_trajectory, skip_replay_buffer_state
from ..td3 import TD3TrainMetric, TD3Workflow


logger = logging.getLogger(__name__)


class WorkflowMetric(MetricBase):
    sampled_timesteps: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_timesteps_per_agent: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class PopTD3Workflow(TD3Workflow):
    """
    indepentent TD3 agent with shared replay buffer
    """

    @classmethod
    def name(cls):
        return "PopTD3"

    def _setup_workflow_metrics(self) -> MetricBase:
        return WorkflowMetric()

    def setup(self, key: chex.PRNGKey) -> State:
        key, agent_key, env_key, rb_key = jax.random.split(key, 4)

        agent_state, opt_state = self._setup_agent_and_optimizer(agent_key)
        workflow_metrics = self._setup_workflow_metrics()

        if self.enable_multi_devices:
            # workflow_metrics, agent_state, opt_state = jax.device_put_replicated(
            #     (workflow_metrics, agent_state, opt_state), self.devices
            # )

            # # key and env_state should be different over devices
            # key = split_key_to_devices(key, self.devices)

            # env_key = split_key_to_devices(env_key, self.devices)
            # env_state = jax.pmap(self.env.reset, axis_name=self.pmap_axis_name)(env_key)
            # rb_key = split_key_to_devices(rb_key, self.devices)
            # replay_buffer_state = jax.pmap(
            #     self._setup_replaybuffer, axis_name=self.pmap_axis_name
            # )(rb_key)
            raise NotImplementedError("multi_devices is not supported yet")
        else:
            # TODO: what about using shared init env_state?
            env_state = jax.vmap(self.env.reset)(
                jax.random.split(env_key, self.config.pop_size)
            )
            replay_buffer_state = self._setup_replaybuffer(rb_key)

        state = State(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state,
            replay_buffer_state=replay_buffer_state,
        )

        logger.info("Start replay buffer post-setup")
        if self.enable_multi_devices:
            state = jax.pmap(
                self._postsetup_replaybuffer, axis_name=self.pmap_axis_name
            )(state)
        else:
            state = self._postsetup_replaybuffer(state)

        logger.info("Complete replay buffer post-setup")

        return state

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree]:
        agent_state = jax.vmap(self.agent.init, in_axes=(None, None, 0))(
            self.env.obs_space,
            self.env.action_space,
            jax.random.split(key, self.config.pop_size),
        )

        def _opt_init(agent_state):
            return PyTreeDict(
                actor=self.optimizer.init(agent_state.params.actor_params),
                critic=self.optimizer.init(agent_state.params.critic_params),
            )

        opt_state = jax.vmap(_opt_init)(agent_state)

        return agent_state, opt_state

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
                        pmap_axis_name=self.pmap_axis_name,
                    )
                )
            return agent_state

        # ==== fill random transitions ====

        key, random_rollout_key, rollout_key = jax.random.split(state.key, num=3)
        random_agent = RandomAgent()
        random_agent_state = random_agent.init(
            obs_space, action_space, jax.random.PRNGKey(0)
        )
        rollout_length = config.random_timesteps // config.num_envs

        trajectory = _rollout(
            random_agent,
            random_agent_state,
            key=random_rollout_key,
            rollout_length=rollout_length,
        )

        agent_state = jax.vmap(_update_obs_preprocessor, in_axes=(0, None))(
            agent_state, trajectory
        )

        replay_buffer_state = self.replay_buffer.add(replay_buffer_state, trajectory)

        rollout_timesteps = rollout_length * config.num_envs
        sampled_timesteps = psum(
            jnp.uint32(rollout_timesteps), axis_name=self.pmap_axis_name
        )

        # ==== fill tansition state from init agents (diff from TD3) ====
        rollout_length = math.ceil(
            (config.learning_start_timesteps - rollout_timesteps)
            / (config.num_envs * config.pop_size)
        )

        _vmap_rollout = jax.vmap(
            partial(_rollout, self.agent, rollout_length=rollout_length)
        )

        trajectory = _vmap_rollout(
            agent_state, jax.random.split(rollout_key, config.pop_size)
        )
        agent_state = jax.vmap(_update_obs_preprocessor)(agent_state, trajectory)

        # [#pop, T*B] -> [#pop*T*B, ...]
        trajectory = flatten_rollout_trajectory(trajectory)
        replay_buffer_state = self.replay_buffer.add(replay_buffer_state, trajectory)

        rollout_timesteps = rollout_length * config.num_envs * config.pop_size
        sampled_timesteps = sampled_timesteps + psum(
            jnp.uint32(rollout_timesteps), axis_name=self.pmap_axis_name
        )

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
        )

    def step(self, state: State) -> tuple[MetricBase, State]:
        """
        the basic step function for the workflow to update agent
        """
        pop_size = self.config.pop_size
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        _rollout = partial(
            rollout,
            self.env.step,
            self.agent.compute_actions,
            rollout_length=self.config.rollout_length,
            env_extra_fields=("ori_obs", "termination"),
        )

        # the trajectory [#pop, T, B, ...]
        trajectory, env_state = jax.vmap(_rollout)(
            state.env_state, state.agent_state, jax.random.split(rollout_key, pop_size)
        )

        trajectory = clean_trajectory(trajectory)
        trajectory = flatten_pop_rollout_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        agent_state = state.agent_state
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

        critic_update_fn = jax.vmap(critic_update_fn, in_axes=(0, 0, None, 0))
        actor_update_fn = jax.vmap(actor_update_fn, in_axes=(0, 0, None, 0))

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            critic_opt_state = opt_state.critic
            actor_opt_state = opt_state.actor

            key, critic_key, actor_key, rb_key = jax.random.split(key, num=4)

            if self.config.actor_update_interval - 1 > 0:

                def _sample_and_update_critic_fn(carry, unused_t):
                    key, agent_state, critic_opt_state = carry

                    key, rb_key, critic_key = jax.random.split(key, num=3)
                    # it's safe to use read-only replay_buffer_state here.
                    sample_batch = self.replay_buffer.sample(
                        replay_buffer_state, rb_key
                    ).experience

                    critic_key = jax.random.split(critic_key, pop_size)

                    (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
                        critic_update_fn(
                            critic_opt_state, agent_state, sample_batch, critic_key
                        )
                    )

                    return (key, agent_state, critic_opt_state), None

                key, critic_multiple_update_key = jax.random.split(key)

                (_, agent_state, critic_opt_state), _ = jax.lax.scan(
                    _sample_and_update_critic_fn,
                    (critic_multiple_update_key, agent_state, critic_opt_state),
                    (),
                    length=self.config.actor_update_interval - 1,
                )

            sample_batch = self.replay_buffer.sample(
                replay_buffer_state, rb_key
            ).experience

            critic_key = jax.random.split(critic_key, pop_size)
            actor_key = jax.random.split(actor_key, pop_size)

            (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
                critic_update_fn(
                    critic_opt_state, agent_state, sample_batch, critic_key
                )
            )

            (actor_loss, actor_loss_dict), agent_state, actor_opt_state = (
                actor_update_fn(actor_opt_state, agent_state, sample_batch, actor_key)
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

            opt_state = opt_state.replace(
                actor=actor_opt_state, critic=critic_opt_state
            )

            return (
                (key, agent_state, opt_state),
                (critic_loss, actor_loss, critic_loss_dict, actor_loss_dict),
            )

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
            (learn_key, agent_state, state.opt_state),
            (),
            length=self.config.num_updates_per_iter,
        )

        # [#pop, ...]
        train_metrics = TD3TrainMetric(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        # calculate the number of timestep
        sampled_timesteps_per_agent = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.pmap_axis_name,
        )
        sampled_timesteps = psum(
            jnp.uint32(
                self.config.rollout_length * self.config.num_envs * self.config.pop_size
            ),
            axis_name=self.pmap_axis_name,
        )

        # iterations is the number of updates of the agent
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_timesteps_per_agent=state.metrics.sampled_timesteps_per_agent
            + sampled_timesteps_per_agent,
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

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        key, eval_key = jax.random.split(state.key, num=2)

        # [#pop, #episodes]
        raw_eval_metrics = self.evaluator.evaluate(
            state.agent_state,
            jax.random.split(eval_key, self.config.pop_size),
            num_episodes=self.config.eval_episodes,
        )

        eval_metrics = EvaluateMetric(
            episode_returns=raw_eval_metrics.episode_returns.mean(-1),
            episode_lengths=raw_eval_metrics.episode_lengths.mean(-1),
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        state = state.replace(key=key)
        return eval_metrics, state

    def learn(self, state: State) -> State:
        num_devices = jax.device_count()
        one_step_timesteps = (
            self.config.rollout_length * self.config.num_envs * self.config.pop_size
        )
        sampled_timesteps = tree_unpmap(state.metrics.sampled_timesteps).tolist()
        num_iters = math.ceil(
            (self.config.total_timesteps - sampled_timesteps)
            / (one_step_timesteps * self.config.fold_iters * num_devices)
        )

        for i in range(num_iters):
            train_metrics, state = self._multi_steps(state)
            workflow_metrics = state.metrics

            # current iteration
            iterations = tree_unpmap(
                state.metrics.iterations, self.pmap_axis_name
            ).tolist()
            train_metrics = tree_unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = tree_unpmap(workflow_metrics, self.pmap_axis_name)
            self.recorder.write(workflow_metrics.to_local_dict(), iterations)

            train_metrics_dict = jtu.tree_map(
                partial(get_1d_array_statistics, histogram=True),
                train_metrics.to_local_dict(),
            )

            self.recorder.write(train_metrics_dict, iterations)

            if iterations % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)
                eval_metrics = tree_unpmap(eval_metrics, self.pmap_axis_name)

                eval_metrics_dict = jtu.tree_map(
                    partial(get_1d_array_statistics, histogram=True),
                    eval_metrics.to_local_dict(),
                )

                self.recorder.write(add_prefix(eval_metrics_dict, "eval"), iterations)

            saved_state = tree_unpmap(state, self.pmap_axis_name)
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)
            self.checkpoint_manager.save(
                iterations,
                args=ocp.args.StandardSave(saved_state),
            )

        return state
