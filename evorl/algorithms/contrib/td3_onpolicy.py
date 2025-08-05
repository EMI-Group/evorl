import logging
from functools import partial
import math

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from omegaconf import DictConfig

from evorl.distributed import psum, unpmap
from evorl.distributed.gradients import agent_gradient_update
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators import Evaluator
from evorl.metrics import MetricBase
from evorl.rollout import rollout
from evorl.types import (
    PyTreeDict,
    State,
)
from evorl.utils import running_statistics
from evorl.utils.jax_utils import scan_and_mean, tree_stop_gradient, scan_and_last
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update
from evorl.agent import Agent, AgentState
from evorl.recorders import add_prefix
from evorl.workflows import OnPolicyWorkflow

from evorl.algorithms.td3 import make_mlp_td3_agent, clean_trajectory, TD3TrainMetric

logger = logging.getLogger(__name__)

MISSING_LOSS = -1e10


class TD3OnPolicyWorkflow(OnPolicyWorkflow):
    @classmethod
    def name(cls):
        return "TD3-OnPolicy"

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        num_devices = jax.device_count()

        if config.num_envs % num_devices != 0:
            logger.warning(
                f"num_envs({config.num_envs}) cannot be divided by num_devices({num_devices}), "
                f"rescale num_envs to {config.num_envs // num_devices}"
            )
        if config.num_eval_envs % num_devices != 0:
            logger.warning(
                f"num_eval_envs({config.num_eval_envs}) cannot be divided by num_devices({num_devices}), "
                f"rescale num_eval_envs to {config.num_eval_envs // num_devices}"
            )
        if config.minibatch_size % num_devices != 0:
            logger.warning(
                f"minibatch_size({config.minibatch_size}) cannot be divided by num_devices({num_devices}), "
                f"rescale minibatch_size to {config.minibatch_size // num_devices}"
            )

        config.num_envs = config.num_envs // num_devices
        config.num_eval_envs = config.num_eval_envs // num_devices
        config.minibatch_size = config.minibatch_size // num_devices

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
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

        eval_env = create_env(
            config.env,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        one_step_rollout_steps = config.num_envs * config.rollout_length
        if one_step_rollout_steps % config.minibatch_size != 0:
            logger.warning(
                f"minibatch_size ({config.minibath_size} cannot divides num_envs*rollout_length)"
            )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        return cls(env, agent, optimizer, evaluator, config)

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree]:
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        opt_state = PyTreeDict(
            actor=self.optimizer.init(agent_state.params.actor_params),
            critic=self.optimizer.init(agent_state.params.critic_params),
        )
        return agent_state, opt_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        # trajectory: [T, #envs, ...]
        trajectory, env_state = rollout(
            self.env.step,
            self.agent.compute_actions,
            state.env_state,
            state.agent_state,
            rollout_key,
            rollout_length=self.config.rollout_length,
            env_extra_fields=("ori_obs", "termination"),
        )

        agent_state = state.agent_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    pmap_axis_name=self.pmap_axis_name,
                )
            )

        trajectory = clean_trajectory(trajectory)
        trajectory = flatten_rollout_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)
        # ============================

        update_fn = build_rl_update_fn(self.agent, self.optimizer, self.config)

        num_minibatches = (
            self.config.rollout_length
            * self.config.num_envs
            // (self.config.minibatch_size * self.config.actor_update_interval)
        )

        def _get_shuffled_minibatch(perm_key, x):
            x = jax.random.permutation(perm_key, x)[
                : num_minibatches
                * self.config.minibatch_size
                * self.config.actor_update_interval
            ]
            return x.reshape(
                num_minibatches,
                self.config.actor_update_interval,
                self.config.minibatch_size,
                *x.shape[1:],
            )

        def minibatch_step(carry, trajectory):
            # trajectory: [actor_update_interval, B, ...]

            opt_state, agent_state, key = carry
            key, learn_key = jax.random.split(key)

            (agent_state, opt_state), train_info = update_fn(
                agent_state, opt_state, trajectory, learn_key
            )

            return (opt_state, agent_state, key), train_info

        def epoch_step(carry, _):
            opt_state, agent_state, key = carry
            perm_key, learn_key = jax.random.split(key, num=2)

            (opt_state, agent_state, key), train_info = scan_and_mean(
                minibatch_step,
                (opt_state, agent_state, learn_key),
                jtu.tree_map(partial(_get_shuffled_minibatch, perm_key), trajectory),
                length=num_minibatches,
            )

            return (opt_state, agent_state, key), train_info

        # loss_list: [reuse_rollout_epochs, num_minibatches]
        (
            (opt_state, agent_state, _),
            (
                critic_loss,
                actor_loss,
                critic_loss_dict,
                actor_loss_dict,
            ),
        ) = scan_and_last(
            epoch_step,
            (state.opt_state, agent_state, learn_key),
            None,
            length=self.config.reuse_rollout_epochs,
        )

        # ======== update metrics ========

        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.pmap_axis_name,
        )

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        train_metrics = TD3TrainMetric(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state,
        )

    def learn(self, state: State) -> State:
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        num_iters = math.ceil(self.config.total_timesteps / one_step_timesteps)

        start_iteration = unpmap(state.metrics.iterations, self.pmap_axis_name).tolist()
        final_iteration = num_iters + start_iteration

        for i in range(num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            iterations = unpmap(state.metrics.iterations, self.pmap_axis_name).tolist()
            train_metrics = unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = unpmap(workflow_metrics, self.pmap_axis_name)

            self.recorder.write(workflow_metrics.to_local_dict(), iterations)
            self.recorder.write(train_metrics.to_local_dict(), iterations)

            if (
                iterations % self.config.eval_interval == 0
                or iterations == final_iteration
            ):
                eval_metrics, state = self.evaluate(state)
                eval_metrics = unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iterations
                )

            self.checkpoint_manager.save(
                iterations,
                unpmap(state, self.pmap_axis_name),
                force=iterations == final_iteration,
            )

        return state


def build_rl_update_fn(
    agent: Agent,
    optimizer: optax.GradientTransformation,
    config: DictConfig,
):
    def critic_loss_fn(agent_state, sample_batch, key):
        # loss on a single critic with multiple actors
        # sample_batch: (B, ...)

        loss_dict = agent.critic_loss(agent_state, sample_batch, key)

        loss = loss_dict.critic_loss

        return loss, loss_dict

    def actor_loss_fn(agent_state, sample_batch, key):
        # loss on a single actor
        # different actor shares same sample_batch (B, ...) input
        loss_dict = agent.actor_loss(agent_state, sample_batch, key)

        loss = loss_dict.actor_loss

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
