from evorl.utils import running_statistics
import logging
from functools import partial
from omegaconf import DictConfig
from flax import nnx

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.agent import AgentState
from evorl.envs import Box, Space, create_env, AutoresetMode
from evorl.evaluators import Evaluator
from evorl.distributed import pmean_nnx_loss_and_grad, psum
from evorl.metrics import TrainMetric, MetricBase
from evorl.rollout import rollout
from evorl.sample_batch import SampleBatch
from evorl.types import LossDict, PyTreeDict, State
from evorl.utils.jax_utils import tree_stop_gradient, scan_and_mean
from evorl.utils.rl_toolkits import (
    average_episode_discount_return,
    compute_gae,
    flatten_rollout_trajectory,
)

from evorl.algorithms.ppo import PPOWorkflow, PPOAgent

logger = logging.getLogger(__name__)


class ParamPPOAgent(PPOAgent):
    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        agent_state = super().init(obs_space, action_space, key)

        return agent_state.replace(
            extra_state=PyTreeDict(clip_epsilon=jnp.float32(self.clip_epsilon))
        )

    def loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
        *,
        axis_name: str | None = None,
    ) -> LossDict:
        return super().loss(
            agent_state,
            sample_batch,
            key,
            clip_epsilon=agent_state.extra_state.clip_epsilon,
            axis_name=axis_name,
        )


class ParamPPOWorkflow(PPOWorkflow):
    @classmethod
    def name(cls):
        return "ParamPPO"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        max_episode_steps = config.env.max_episode_steps

        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.ENVPOOL,
        )

        agent = ParamPPOAgent(
            agent_network_cfg=config.agent_network,
            continuous_action=isinstance(env.action_space, Box),
            clip_epsilon=config.clip_epsilon,
            normalize_gae=config.normalize_gae,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
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
            max_episode_steps=max_episode_steps,
        )

        return cls(env, agent, optimizer, evaluator, config)

    def setup(self, key: jax.Array) -> State:
        state = super().setup(key)

        return state.replace(
            hp_state=PyTreeDict(
                gae_lambda_g=-jnp.log(1 - jnp.float32(self.config.gae_lambda)),
                discount_g=-jnp.log(
                    1 - jnp.float32(self.config.discount)
                ),  # discount = 1 - exp(-g)
                actor_loss_weight=jnp.float32(self.config.loss_weights.actor_loss),
                critic_loss_weight=jnp.float32(self.config.loss_weights.critic_loss),
                entropy_loss_weight=jnp.float32(self.config.loss_weights.actor_entropy),
            )
        )

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)
        if self.is_multi_device:
            rank = jax.lax.axis_index(self.dp_axis_name)
            rollout_key = jax.random.fold_in(rollout_key, rank)
            learn_key = jax.random.fold_in(learn_key, rank)

        # trajectory: [T, #envs, ...]
        trajectory, env_state = rollout(
            self.env.step,
            self.agent.compute_actions,
            state.env_state,
            state.agent_state,
            rollout_key,
            rollout_length=self.config.rollout_length,
            env_extra_fields=("autoreset", "episode_return", "termination"),
        )

        agent_state = state.agent_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    axis_name=self.dp_axis_name,
                )
            )

        train_episode_return = average_episode_discount_return(
            trajectory.extras.env_extras.episode_return,
            trajectory.dones,
            axis_name=self.dp_axis_name,
        )

        # ======== compute GAE =======
        _obs = jtu.tree_map(
            lambda obs, next_obs: jnp.concatenate([obs, next_obs[-1:]], axis=0),
            trajectory.obs,
            trajectory.next_obs,
        )
        # concat [values, bootstrap_value]
        vs = self.agent.compute_values(state.agent_state, SampleBatch(obs=_obs))

        gae_lambda = 1 - jnp.exp(-state.hp_state.gae_lambda_g)
        discount = 1 - jnp.exp(-state.hp_state.discount_g)

        v_targets, advantages = compute_gae(
            rewards=trajectory.rewards,  # peb_rewards
            values=vs,
            dones=trajectory.dones,
            terminations=trajectory.extras.env_extras.termination,
            gae_lambda=gae_lambda,
            discount=discount,
        )
        trajectory.extras.v_targets = jax.lax.stop_gradient(v_targets)
        trajectory.extras.advantages = jax.lax.stop_gradient(advantages)
        # [T,B,...] -> [T*B,...]
        trajectory = tree_stop_gradient(flatten_rollout_trajectory(trajectory))
        # ============================

        def loss_fn(models, agent_state, sample_batch, key):
            # learn all data from trajectory
            loss_dict = self.agent.loss(
                agent_state.replace(model_state=nnx.unpack(models)),
                sample_batch,
                key,
                axis_name=self.dp_axis_name,
            )
            loss_weights = dict(
                actor_loss=state.hp_state.actor_loss_weight,
                critic_loss=state.hp_state.critic_loss_weight,
                actor_entropy=state.hp_state.entropy_loss_weight,
            )
            loss = jnp.zeros(())
            for loss_key, weight in loss_weights.items():
                loss += weight * loss_dict[loss_key]

            return loss, loss_dict

        loss_and_grad = pmean_nnx_loss_and_grad(
            loss_fn, self.dp_axis_name, has_aux=True
        )

        num_minibatches = (
            self.config.rollout_length
            * self.config.num_envs
            // self.config.minibatch_size
        )

        def _get_shuffled_minibatch(perm_key, x):
            x = x[jax.random.permutation(perm_key, x.shape[0])][
                : num_minibatches * self.config.minibatch_size
            ]
            return x.reshape(num_minibatches, self.config.minibatch_size, *x.shape[1:])

        def minibatch_step(carry, trajectory):
            opt_state, agent_state, key = carry
            key, learn_key = jax.random.split(key)

            models = nnx.merge(agent_state.model_state, copy=True)
            optimizer = nnx.merge(opt_state, copy=True)
            (loss, loss_dict), grads = loss_and_grad(
                models, agent_state, trajectory, learn_key
            )
            optimizer.update(models, grads)
            agent_state = agent_state.replace(model_state=nnx.unpack(models))
            opt_state = nnx.unpack(optimizer)

            return (opt_state, agent_state, key), (loss, loss_dict)

        def epoch_step(carry, _):
            opt_state, agent_state, key = carry
            perm_key, learn_key = jax.random.split(key, num=2)

            (opt_state, agent_state, key), (loss, loss_dict) = scan_and_mean(
                minibatch_step,
                (opt_state, agent_state, learn_key),
                jtu.tree_map(partial(_get_shuffled_minibatch, perm_key), trajectory),
                length=num_minibatches,
            )

            return (opt_state, agent_state, key), (loss, loss_dict)

        # loss_list: [reuse_rollout_epochs, num_minibatches]
        (opt_state, agent_state, _), (loss, loss_dict) = scan_and_mean(
            epoch_step,
            (state.opt_state, agent_state, learn_key),
            None,
            length=self.config.reuse_rollout_epochs,
        )

        # ======== update metrics ========

        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.dp_axis_name,
        )
        sampled_epsiodes = psum(
            trajectory.dones.sum().astype(jnp.uint32), axis_name=self.dp_axis_name
        )

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_epsiodes,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(axis_name=self.dp_axis_name)

        train_metrics = TrainMetric(
            train_episode_return=train_episode_return,
            loss=loss,
            raw_loss_dict=loss_dict,
        ).all_reduce(axis_name=self.dp_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state,
        )
