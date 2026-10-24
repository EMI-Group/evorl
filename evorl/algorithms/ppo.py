from evorl.utils import running_statistics
import logging
import math
from functools import partial
from typing import Any
from omegaconf import DictConfig

from flax import nnx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.distributed import pmean, pmean_nnx_loss_and_grad, psum
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.envs import Discrete, Box, AutoresetMode, create_env, Space
from evorl.evaluators import Evaluator
from evorl.metrics import TrainMetric, MetricBase
from evorl.networks import make_policy_network, make_v_network
from evorl.rollout import rollout
from evorl.sample_batch import SampleBatch
from evorl.types import (
    MISSING_REWARD,
    Action,
    LossDict,
    PolicyExtraInfo,
    PyTreeDict,
    State,
    pytree_field,
)
from evorl.utils.jax_utils import tree_stop_gradient, scan_and_mean, rng_fold_in_axis
from evorl.utils.rl_toolkits import (
    average_episode_discount_return,
    compute_gae_with_horizon,
    flatten_rollout_trajectory,
    approximate_kl,
)
from evorl.workflows import OnPolicyWorkflow
from evorl.recorders import add_prefix
from evorl.agent import Agent, AgentState

logger = logging.getLogger(__name__)


class PPOAgent(Agent):
    continuous_action: bool
    agent_network_cfg: DictConfig = pytree_field(static=True)
    obs_preprocessor: Any = pytree_field(default=None, static=True)

    clip_epsilon: float = 0.2
    normalize_gae: bool = True

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        if not isinstance(action_space, (Box, Discrete)):
            raise TypeError(f"Unsupported action space: {action_space}")
        policy_key, value_key = jax.random.split(key, 2)

        network_cfg = self.agent_network_cfg
        policy_obs_key = network_cfg.get("policy_obs_key")
        value_obs_key = network_cfg.get("value_obs_key")
        policy_space = obs_space.spaces[policy_obs_key] if policy_obs_key else obs_space
        value_space = obs_space.spaces[value_obs_key] if value_obs_key else obs_space
        action_size = (
            action_space.shape[0] * 2 if self.continuous_action else action_space.n
        )
        models = nnx.Dict(
            policy=make_policy_network(
                input_size=policy_space.shape[-1],
                action_size=action_size,
                hidden_layer_sizes=tuple(network_cfg.actor_hidden_layer_sizes),
                obs_key=policy_obs_key,
                rngs=nnx.Rngs(policy_key),
            ),
            value=make_v_network(
                input_size=value_space.shape[-1],
                hidden_layer_sizes=tuple(network_cfg.critic_hidden_layer_sizes),
                obs_key=value_obs_key,
                rngs=nnx.Rngs(value_key),
            ),
        )

        if self.normalize_obs:
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(
                obs_space.sample(key)
            )
        else:
            obs_preprocessor_state = None

        return AgentState(
            model_state=nnx.unpack(models),
            obs_preprocessor_state=obs_preprocessor_state,
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.get_models(agent_state, copy=True).policy(obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.sample(seed=key)

        policy_extras = PyTreeDict(
            # raw_action=raw_actions,
            logp=actions_dist.log_prob(actions)
        )

        return actions, policy_extras

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: jax.Array
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.get_models(agent_state, copy=True).policy(obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.mode()

        return actions, PyTreeDict()

    def loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
        clip_epsilon: jax.Array | None = None,
        *,
        axis_name: str | None = None,
    ) -> LossDict:
        """Compute PPO losses using valid samples across one policy's devices.

        Args:
            agent_state: Functional model and observation-preprocessor state.
            sample_batch: Local minibatch, including the autoreset mask.
            key: Random key for entropy estimation.
            clip_epsilon: Optional override of the policy clipping threshold.
            axis_name: Data-parallel Manual axis, or `None` for local training.
                Independent population members must not share this axis.

        Returns:
            Loss terms weighted for the subsequent device mean. Advantage
            statistics exclude autoreset samples and span the global minibatch.
            An empty minibatch contributes zero actor, critic and entropy loss.
        """
        models = self.get_models(agent_state)
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        # mask invalid transitions at autoreset
        mask = jnp.logical_not(sample_batch.extras.env_extras.autoreset)
        # A mean of local sums divided by this denominator is the global valid
        # sample mean, including when one device has no valid samples.
        mean_count = pmean(mask.sum(dtype=jnp.float32), axis_name)
        denominator = jnp.where(mean_count > 0, mean_count, 1.0)

        # ======= critic =======
        vs = models.value(obs)

        v_targets = sample_batch.extras.v_targets

        critic_loss = optax.squared_error(vs, v_targets).sum(where=mask) / denominator

        # ====== actor =======

        # [T*B, A]
        raw_actions = models.policy(obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        # [T*B]
        actions_logp = actions_dist.log_prob(sample_batch.actions)
        behavior_actions_logp = sample_batch.extras.policy_extras.logp

        advantages = sample_batch.extras.advantages
        if self.normalize_gae:
            mean = pmean(advantages.sum(where=mask), axis_name) / denominator
            variance = (
                pmean(jnp.square(advantages - mean).sum(where=mask), axis_name)
                / denominator
            )
            advantages = (advantages - mean) / (jnp.sqrt(variance) + 1e-8)

        if clip_epsilon is None:
            clip_epsilon = self.clip_epsilon

        logrho = actions_logp - behavior_actions_logp
        rho = jnp.exp(logrho)

        # advantages: [T*B]
        policy_sorrogate_loss1 = rho * advantages
        policy_sorrogate_loss2 = (
            jnp.clip(rho, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        )
        actor_loss = (
            -jnp.minimum(policy_sorrogate_loss1, policy_sorrogate_loss2).sum(where=mask)
            / denominator
        )

        # entropy: [T*B]
        if self.continuous_action:
            actor_entropy = actions_dist.entropy(seed=key).sum(where=mask) / denominator
        else:
            actor_entropy = actions_dist.entropy().sum(where=mask) / denominator

        approx_kl = approximate_kl(logrho, axis=()).sum(where=mask) / denominator

        return PyTreeDict(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            actor_entropy=actor_entropy,
            approx_kl=approx_kl,
        )

    def compute_values(
        self, agent_state: AgentState, sample_batch: SampleBatch
    ) -> jax.Array:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        return self.get_models(agent_state, copy=True).value(obs)


class PPOWorkflow(OnPolicyWorkflow):
    @classmethod
    def name(cls):
        return "PPO"

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        super()._rescale_config(config)
        num_devices = jax.local_device_count()

        for name in ("num_envs", "minibatch_size"):
            value = config[name]
            if value < num_devices:
                raise ValueError(
                    f"{name}({value}) must be at least num_devices({num_devices})"
                )

        if config.num_envs % num_devices != 0:
            logger.warning(
                f"num_envs({config.num_envs}) cannot be divided by num_devices({num_devices}), "
                f"rescale num_envs to {config.num_envs // num_devices}"
            )
        if config.minibatch_size % num_devices != 0:
            logger.warning(
                f"minibatch_size({config.minibatch_size}) cannot be divided by num_devices({num_devices}), "
                f"rescale minibatch_size to {config.minibatch_size // num_devices}"
            )

        config.num_envs = config.num_envs // num_devices
        config.minibatch_size = config.minibatch_size // num_devices

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        max_episode_steps = config.env.max_episode_steps

        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.ENVPOOL,
        )

        agent = PPOAgent(
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

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)
        rollout_key, learn_key = rng_fold_in_axis(
            (rollout_key, learn_key), self.dp_axis_name
        )

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

        v_targets, advantages = compute_gae_with_horizon(
            rewards=trajectory.rewards,
            values=vs,
            dones=trajectory.dones,
            terminations=trajectory.extras.env_extras.termination,
            gae_horizon=self.config.gae_horizon,
            gae_lambda=self.config.gae_lambda,
            discount=self.config.discount,
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
            loss_weights = self.config.loss_weights
            loss = jnp.zeros(())
            for loss_key in loss_weights:
                loss += loss_weights[loss_key] * loss_dict[loss_key]

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

    def learn(self, state: State) -> State:
        one_step_timesteps = (
            self.config.rollout_length * self.config.num_envs * self.num_devices
        )
        num_iters = math.ceil(self.config.total_timesteps / one_step_timesteps)

        start_iteration = state.metrics.iterations

        for i in range(start_iteration, num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            iters = i + 1

            self.recorder.write(workflow_metrics.to_local_dict(), iters)
            train_metric_data = train_metrics.to_local_dict()
            if train_metrics.train_episode_return == MISSING_REWARD:
                train_metric_data["train_episode_return"] = None
            self.recorder.write(train_metric_data, iters)

            if iters % self.config.eval_interval == 0 or iters == num_iters:
                eval_metrics, state = self.evaluate(state)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iters
                )

            self.checkpoint_manager.save(
                iters,
                state,
                force=iters == num_iters,
            )

        return state
