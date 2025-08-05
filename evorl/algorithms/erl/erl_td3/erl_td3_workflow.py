import logging
from omegaconf import DictConfig
import math

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.agent import AgentState, Agent
from evorl.distributed import agent_gradient_update
from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import (
    right_shift_with_padding,
    tree_stop_gradient,
    scan_and_last,
    scan_and_mean,
)
from evorl.utils.rl_toolkits import (
    flatten_rollout_trajectory,
    flatten_pop_rollout_episode,
    soft_target_update,
)
from evorl.recorders import get_1d_array_statistics
from evorl.algorithms.td3 import TD3TrainMetric, TD3NetworkParams

from ..erl_workflow import ERLWorkflowBase, ERLTrainMetric

logger = logging.getLogger(__name__)


class ERLTD3WorkflowTemplate(ERLWorkflowBase):
    """A template for ERL workflow on TD3 Agent."""

    # Note: Turn off the warmup logging in PBT or parallel training
    LOGGING_WARMUP_FLAG = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rl_update_fn = build_erl_rl_update_fn(
            self.agent,
            self.optimizer,
            self.config,
            self.agent_state_vmap_axes,
        )

    def setup(self, key: chex.PRNGKey) -> State:
        state = super().setup(key)

        # Note: we assume
        if self.config.warmup_iters > 0:
            logger.info("Start warmup")

            def _warmup_step(state, unused_t):
                train_metrics, state = self.warmup_step(state)
                return state, train_metrics

            def _logging(train_metrics, iters):
                if self.LOGGING_WARMUP_FLAG:
                    train_metrics_dict = train_metrics.to_local_dict()
                    train_metrics_dict["pop_episode_returns"] = get_1d_array_statistics(
                        train_metrics_dict["pop_episode_returns"], histogram=True
                    )

                    train_metrics_dict["pop_episode_lengths"] = get_1d_array_statistics(
                        train_metrics_dict["pop_episode_lengths"], histogram=True
                    )
                    del train_metrics_dict["rl_episode_lengths"]
                    del train_metrics_dict["rl_episode_returns"]
                    del train_metrics_dict["rl_metrics"]
                    self.recorder.write(train_metrics_dict, state.metrics.iterations)

            num_fold_iters = math.floor(
                self.config.warmup_iters / self.config.eval_interval
            )
            last_fold_iters = self.config.warmup_iters % self.config.eval_interval

            for i in range(num_fold_iters):
                state, train_metrics = scan_and_last(
                    _warmup_step, state, (), length=self.config.eval_interval
                )
                _logging(train_metrics, state.metrics.iterations)

            if last_fold_iters > 0:
                state, train_metrics = scan_and_last(
                    _warmup_step, state, (), length=last_fold_iters
                )
                _logging(train_metrics, state.metrics.iterations)

            logger.info("Complete warmup")

        return state

    def warmup_step(self, state: State) -> tuple[ERLTrainMetric, State]:
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        ec_opt_state = state.ec_opt_state
        replay_buffer_state = state.replay_buffer_state

        key, ec_rollout_key = jax.random.split(state.key, 2)

        # 1. ask()
        pop_actor_params, ec_opt_state = self.ec_optimizer.ask(ec_opt_state)
        # 2. evaluate()
        pop_agent_state = erl_replace_td3_actor_params(agent_state, pop_actor_params)
        ec_eval_metrics, ec_trajectory, replay_buffer_state = self._ec_rollout(
            pop_agent_state, replay_buffer_state, ec_rollout_key
        )
        fitnesses = ec_eval_metrics.episode_returns.mean(axis=-1)
        # 3. tell()
        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        train_metrics = ERLTrainMetric(
            pop_episode_lengths=ec_eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=ec_eval_metrics.episode_returns.mean(-1),
            ec_info=ec_metrics,
            rb_size=replay_buffer_state.buffer_size,
        )

        # calculate the number of timestep
        sampled_timesteps = ec_eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        sampled_episodes = jnp.uint32(self.config.episodes_for_fitness * pop_size)

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
            iterations=state.metrics.iterations + 1,
        )

        state = state.replace(
            key=key,
            metrics=workflow_metrics,
            replay_buffer_state=replay_buffer_state,
            ec_opt_state=ec_opt_state,
        )

        return train_metrics, state

    def _ec_rollout(self, agent_state, replay_buffer_state, key):
        return rollout_episode(
            agent_state,
            replay_buffer_state,
            key,
            collector=self.ec_collector,
            replay_buffer=self.replay_buffer,
            agent_state_vmap_axes=self.agent_state_vmap_axes,
            num_agents=self.config.pop_size,
            num_episodes=self.config.episodes_for_fitness,
        )

    def _rl_rollout(self, agent_state, replay_buffer_state, key):
        return rollout_episode(
            agent_state,
            replay_buffer_state,
            key,
            collector=self.rl_collector,
            replay_buffer=self.replay_buffer,
            agent_state_vmap_axes=self.agent_state_vmap_axes,
            num_agents=self.config.num_rl_agents,
            num_episodes=self.config.rollout_episodes,
        )

    def _rl_update(self, agent_state, opt_state, replay_buffer_state, key):
        def _sample_fn(key):
            return self.replay_buffer.sample(replay_buffer_state, key)

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            key, rb_key, learn_key = jax.random.split(key, 3)

            rb_keys = jax.random.split(
                rb_key, self.config.actor_update_interval * self.config.num_rl_agents
            )
            sample_batches = jax.vmap(_sample_fn)(rb_keys)

            # (actor_update_interval, num_rl_agents, B, ...)
            sample_batches = jtu.tree_map(
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


def erl_replace_td3_actor_params(
    agent_state: AgentState, pop_actor_params: TD3NetworkParams
) -> AgentState:
    return agent_state.replace(
        params=TD3NetworkParams(
            actor_params=pop_actor_params,
            target_actor_params=pop_actor_params,
            critic_params=None,
            target_critic_params=None,
        )
    )


DUMMY_TD3_TRAINMETRIC = TD3TrainMetric(
    critic_loss=jnp.zeros(()),
    actor_loss=jnp.zeros(()),
    raw_loss_dict=PyTreeDict(
        critic_loss=jnp.zeros(()),
        q_value=jnp.zeros(()),
        actor_loss=jnp.zeros(()),
    ),
)


def create_dummy_td3_trainmetric(num: int) -> TD3TrainMetric:
    if num >= 1:
        return DUMMY_TD3_TRAINMETRIC.replace(
            raw_loss_dict=jtu.tree_map(
                lambda x: jnp.broadcast_to(x, (num, *x.shape)),
                DUMMY_TD3_TRAINMETRIC.raw_loss_dict,
            )
        )
    else:
        raise ValueError(f"num should be positive, got {num}")


def rollout_episode(
    agent_state: AgentState,
    replay_buffer_state,
    key,
    *,
    collector,
    replay_buffer,
    agent_state_vmap_axes,
    num_episodes,
    num_agents,
):
    eval_metrics, trajectory = jax.vmap(
        collector.rollout,
        in_axes=(agent_state_vmap_axes, 0, None),
    )(
        agent_state,
        jax.random.split(key, num_agents),
        num_episodes,
    )

    # [n, T, B, ...] -> [T, n*B, ...]
    trajectory = trajectory.replace(next_obs=None)
    trajectory = flatten_pop_rollout_episode(trajectory)

    mask = jnp.logical_not(right_shift_with_padding(trajectory.dones, 1))
    trajectory = trajectory.replace(dones=None)
    trajectory, mask = tree_stop_gradient(
        flatten_rollout_trajectory((trajectory, mask))
    )
    replay_buffer_state = replay_buffer.add(replay_buffer_state, trajectory, mask)

    return eval_metrics, trajectory, replay_buffer_state


def build_erl_rl_update_fn(
    agent: Agent,
    optimizer: optax.GradientTransformation,
    config: DictConfig,
    agent_state_vmap_axes: AgentState,
):
    """K (actor, critic) pairs."""
    num_rl_agents = config.num_rl_agents

    def critic_loss_fn(agent_state, sample_batch, key):
        # loss on a single critic with multiple actors
        # sample_batch: (n, B, ...)

        loss_dict = jax.vmap(agent.critic_loss, in_axes=(agent_state_vmap_axes, 0, 0))(
            agent_state, sample_batch, jax.random.split(key, num_rl_agents)
        )

        loss = loss_dict.critic_loss.sum()

        return loss, loss_dict

    def actor_loss_fn(agent_state, sample_batch, key):
        # loss on a single actor
        loss_dict = jax.vmap(agent.actor_loss, in_axes=(agent_state_vmap_axes, 0, 0))(
            agent_state, sample_batch, jax.random.split(key, num_rl_agents)
        )

        loss = loss_dict.actor_loss.sum()

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

        critic_sample_batches = jtu.tree_map(lambda x: x[:-1], sample_batches)
        last_sample_batch = jtu.tree_map(lambda x: x[-1], sample_batches)

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
