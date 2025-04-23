import logging
import math
from functools import partial
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from omegaconf import DictConfig

from evorl.distributed import agent_gradient_update, psum, unpmap
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.envs import AutoresetMode, create_env, Space, Box, Discrete
from evorl.evaluators import Evaluator
from evorl.metrics import TrainMetric, MetricBase
from evorl.networks import make_policy_network, make_v_network
from evorl.rollout import rollout
from evorl.sample_batch import SampleBatch
from evorl.types import (
    MISSING_REWARD,
    Action,
    LossDict,
    Params,
    PolicyExtraInfo,
    PyTreeData,
    PyTreeDict,
    State,
    pytree_field,
)
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient, scan_and_mean, tree_get
from evorl.utils.rl_toolkits import average_episode_discount_return, approximate_kl
from evorl.workflows import OnPolicyWorkflow
from evorl.agent import Agent, AgentState
from evorl.recorders import add_prefix


logger = logging.getLogger(__name__)


class IMPALANetworkParams(PyTreeData):
    """Contains training state for the learner."""

    policy_params: Params
    value_params: Params


# class IMPALATrainMetric(TrainMetric):
#     rho: chex.Array = jnp.zeros((), dtype=jnp.float32)


class IMPALAAgent(Agent):
    continuous_action: bool
    policy_network: nn.Module
    value_network: nn.Module
    obs_preprocessor: Any = pytree_field(default=None, static=True)

    discount: float = 0.99
    vtrace_lambda: float = 1.0
    clip_rho_threshold: float = 1.0
    clip_c_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
    adv_mode: str = pytree_field(default="official", static=True)

    @property
    def normalize_obs(self):
        return self.obs_preprocessor is not None

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        policy_key, value_key = jax.random.split(key, 2)

        dummy_obs = jtu.tree_map(lambda x: x[None, ...], obs_space.sample(key))

        policy_params = self.policy_network.init(policy_key, dummy_obs)

        value_params = self.value_network.init(value_key, dummy_obs)

        params_state = IMPALANetworkParams(
            policy_params=policy_params, value_params=value_params
        )

        if self.normalize_obs:
            # Note: statistics are broadcasted to [T*B]
            obs_preprocessor_state = running_statistics.init_state(
                tree_get(dummy_obs, 0)
            )
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state, obs_preprocessor_state=obs_preprocessor_state
        )

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.sample(seed=key)

        policy_extras = PyTreeDict(
            # Log probabilities of the selected actions for importance sampling
            logp=actions_dist.log_prob(actions)
        )

        return actions, policy_extras

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        actions = actions_dist.mode()

        return actions, PyTreeDict()

    def loss(
        self, agent_state: AgentState, trajectory: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        """IMPALA loss.

        Args:
            trajectory: [T, B, ...]
                a sequence of transitions, not shuffled timesteps

        """
        # mask invalid transitions at autoreset
        mask = jnp.logical_not(trajectory.extras.env_extras.autoreset)

        obs = trajectory.obs
        _obs = jtu.tree_map(
            lambda obs, next_obs: jnp.concatenate([obs, next_obs[-1:]], axis=0),
            trajectory.obs,
            trajectory.next_obs,
        )
        if self.normalize_obs:
            _obs = self.obs_preprocessor(_obs, agent_state.obs_preprocessor_state)

        vs = self.value_network.apply(agent_state.params.value_params, _obs)

        behavior_actions_logp = trajectory.extras.policy_extras.logp
        behavior_actions = trajectory.actions

        # [T, B, A]
        raw_actions = self.policy_network.apply(agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = get_categorical_dist(raw_actions)

        # [T, B]
        actions_logp = actions_dist.log_prob(behavior_actions)
        logrho = actions_logp - behavior_actions_logp
        rho = jnp.exp(logrho)

        # TODO: consider PEB: truncation in the middle of trajectory
        # hint: use IS of td-error with
        vtrace = compute_vtrace(
            rho_t=rho,
            v_t=vs[:-1],
            v_t_plus_1=vs[1:],
            rewards=trajectory.rewards,
            dones=trajectory.dones,
            terminations=trajectory.extras.env_extras.termination,
            discount=self.discount,
            lambda_=self.vtrace_lambda,
            clip_rho_threshold=self.clip_rho_threshold,
            clip_c_threshold=self.clip_c_threshold,
        )

        vtrace = jax.lax.stop_gradient(vtrace)

        # ======= critic =======

        critic_loss = optax.squared_error(vs[:-1], vtrace).mean(where=mask)

        # ====== actor =======

        # GAE-V: [T*B]
        pg_advantages = compute_pg_advantage(
            vtrace=vtrace,
            v_t=vs[:-1],
            v_t_plus_1=vs[1:],
            rewards=trajectory.rewards,
            terminations=trajectory.extras.env_extras.termination,
            discount=self.discount,
            lambda_=self.vtrace_lambda,
            mode=self.adv_mode,
        )

        clipped_pg_rho_t = jnp.minimum(self.clip_pg_rho_threshold, rho)
        pg_advantage = clipped_pg_rho_t * pg_advantages
        pg_advantage = jax.lax.stop_gradient(pg_advantage)

        policy_loss = -(pg_advantage * actions_logp).mean(where=mask)

        # entropy: [T*B]
        if self.continuous_action:
            actor_entropy = actions_dist.entropy(seed=key).mean(where=mask)
        else:
            actor_entropy = actions_dist.entropy().mean(where=mask)

        approx_kl = approximate_kl(logrho).mean()

        return PyTreeDict(
            actor_loss=policy_loss,
            critic_loss=critic_loss,
            actor_entropy=actor_entropy,
            rho=rho.mean(where=mask),
            approx_kl=approx_kl,
        )


def make_mlp_impala_agent(
    action_space: Space,
    discount: float = 0.99,
    vtrace_lambda: float = 1.0,
    clip_rho_threshold: float = 1.0,
    clip_c_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    adv_mode: str = "official",
    actor_hidden_layer_sizes: tuple[int] = (256, 256),
    critic_hidden_layer_sizes: tuple[int] = (256, 256),
    normalize_obs: bool = False,
    policy_obs_key: str = "",
    value_obs_key: str = "",
):
    if isinstance(action_space, Box):
        action_size = action_space.shape[0] * 2
        continuous_action = True
    elif isinstance(action_space, Discrete):
        action_size = action_space.n
        continuous_action = False
    else:
        raise NotImplementedError(f"Unsupported action space: {action_space}")

    policy_network = make_policy_network(
        action_size=action_size,
        hidden_layer_sizes=actor_hidden_layer_sizes,
        obs_key=policy_obs_key,
    )

    value_network = make_v_network(
        hidden_layer_sizes=critic_hidden_layer_sizes,
        obs_key=value_obs_key,
    )

    if normalize_obs:
        obs_preprocessor = running_statistics.normalize
    else:
        obs_preprocessor = None

    return IMPALAAgent(
        continuous_action=continuous_action,
        policy_network=policy_network,
        value_network=value_network,
        obs_preprocessor=obs_preprocessor,
        discount=discount,
        vtrace_lambda=vtrace_lambda,
        clip_rho_threshold=clip_rho_threshold,
        clip_c_threshold=clip_c_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        adv_mode=adv_mode,
    )


class IMPALAWorkflow(OnPolicyWorkflow):
    """Syncrhonous version of IMPALA (A2C|PPO w/ V-Trace)."""

    @classmethod
    def name(cls):
        return "IMPALA"

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
        max_episode_steps = config.env.max_episode_steps

        env = create_env(
            config.env,
            episode_length=max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.ENVPOOL,
        )

        # Maybe need a discount array for different agents
        agent = make_mlp_impala_agent(
            action_space=env.action_space,
            discount=config.discount,
            vtrace_lambda=config.vtrace_lambda,
            clip_rho_threshold=config.clip_rho_threshold,
            clip_c_threshold=config.clip_c_threshold,
            clip_pg_rho_threshold=config.clip_pg_rho_threshold,
            adv_mode=config.adv_mode,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            normalize_obs=config.normalize_obs,
            policy_obs_key=config.agent_network.policy_obs_key,
            value_obs_key=config.agent_network.value_obs_key,
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
            episode_length=max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=max_episode_steps,
        )

        return cls(env, agent, optimizer, evaluator, config)

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

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
                    pmap_axis_name=self.pmap_axis_name,
                )
            )

        train_episode_return = average_episode_discount_return(
            trajectory.extras.env_extras.episode_return,
            trajectory.dones,
            pmap_axis_name=self.pmap_axis_name,
        )

        trajectory = tree_stop_gradient(trajectory)

        def loss_fn(agent_state, sample_batch, key):
            # learn all data from trajectory
            loss_dict = self.agent.loss(agent_state, sample_batch, key)
            loss_weights = self.config.loss_weights
            loss = jnp.zeros(())
            for loss_key in loss_weights.keys():
                loss += loss_weights[loss_key] * loss_dict[loss_key]

            return loss, loss_dict

        update_fn = agent_gradient_update(
            loss_fn, self.optimizer, pmap_axis_name=self.pmap_axis_name, has_aux=True
        )

        # minibatch_size: num of envs in one batch
        # unit in batch: trajectory [T, B//k, ...]
        num_minibatches = self.config.num_envs // self.config.minibatch_size

        def _get_shuffled_minibatch(perm_key, x):
            # x: [T, B, ...] -> [k, T, B//k, ...]
            x = jax.random.permutation(perm_key, x, axis=1)[
                :, : num_minibatches * self.config.minibatch_size
            ]
            xs = jnp.stack(jnp.split(x, num_minibatches, axis=1))

            return xs

        def minibatch_step(carry, trajectory):
            opt_state, agent_state, key = carry
            key, learn_key = jax.random.split(key)

            (loss, loss_dict), agent_state, opt_state = update_fn(
                opt_state, agent_state, trajectory, learn_key
            )

            return (opt_state, agent_state, key), (loss, loss_dict)

        def epoch_step(carry, _):
            opt_state, agent_state, key = carry
            shuffle_key, learn_key = jax.random.split(key)
            batch_trajectory = jtu.tree_map(
                partial(_get_shuffled_minibatch, shuffle_key), trajectory
            )

            (opt_state, agent_state, key), (loss, loss_dict) = scan_and_mean(
                minibatch_step,
                (opt_state, agent_state, learn_key),
                batch_trajectory,
                length=num_minibatches,
            )

            return (opt_state, agent_state, key), (loss, loss_dict)

        (opt_state, agent_state, _), (loss, loss_dict) = scan_and_mean(
            epoch_step,
            (state.opt_state, agent_state, learn_key),
            None,
            length=self.config.reuse_rollout_epochs,
        )

        # ======== update metrics ========

        sampled_timesteps = psum(
            jnp.array(
                self.config.rollout_length * self.config.num_envs, dtype=jnp.uint32
            ),
            axis_name=self.pmap_axis_name,
        )

        sampled_epsiodes = psum(
            trajectory.dones.sum().astype(jnp.uint32), axis_name=self.pmap_axis_name
        )

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_epsiodes,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        train_metrics = TrainMetric(
            train_episode_return=train_episode_return,
            loss=loss,
            raw_loss_dict=loss_dict,
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

        start_iteration = unpmap(state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            iters = i + 1
            train_metrics = unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = unpmap(workflow_metrics, self.pmap_axis_name)

            self.recorder.write(workflow_metrics.to_local_dict(), iters)
            train_metric_data = train_metrics.to_local_dict()
            if train_metrics.train_episode_return == MISSING_REWARD:
                train_metric_data["train_episode_return"] = None
            self.recorder.write(train_metric_data, iters)

            if iters % self.config.eval_interval == 0 or iters == num_iters:
                eval_metrics, state = self.evaluate(state)
                eval_metrics = unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iters
                )

            self.checkpoint_manager.save(
                iters,
                unpmap(state, self.pmap_axis_name),
                force=iters == num_iters,
            )

        return state


def compute_vtrace(
    rho_t,
    v_t,
    v_t_plus_1,
    rewards,
    dones,
    terminations,
    discount=0.99,
    lambda_=1.0,
    clip_rho_threshold=1.0,
    clip_c_threshold=1.0,
):
    chex.assert_trees_all_equal_shapes_and_dtypes(
        rho_t, v_t, v_t_plus_1, rewards, dones
    )

    # clip c and rho
    clipped_c_t = jnp.minimum(clip_c_threshold, rho_t) * lambda_
    clipped_rho_t = jnp.minimum(clip_rho_threshold, rho_t)

    # calculate δV_t
    td_error = clipped_rho_t * (
        rewards + discount * (1 - terminations) * v_t_plus_1 - v_t
    )

    # calculate delta = vtrace - v_t
    def _compute_delta(delta, params):
        td_error, discount, c = params
        delta = td_error + discount * c * delta
        return delta, delta

    bootstrap_delta = jnp.zeros_like(v_t[-1])
    _, delta = jax.lax.scan(
        _compute_delta,
        bootstrap_delta,
        (td_error, discount * (1 - dones), clipped_c_t),
        reverse=True,
        unroll=16,
    )

    # calculate vs
    vtrace = delta + v_t

    return vtrace


def compute_pg_advantage(
    vtrace,
    v_t,
    v_t_plus_1,
    rewards,
    terminations,
    discount=0.99,
    lambda_=1.0,
    mode="official",
):
    discounts = discount * (1 - terminations)
    # calculate advantage function
    if mode == "official":
        # Note: rllib also follows this implementation
        gae_v_t_plus_1 = jnp.concatenate([vtrace[1:], v_t_plus_1[-1:]], axis=0)
    elif mode == "acme":
        gae_v_t_plus_1 = jnp.concatenate(
            [lambda_ * vtrace[1:] + (1 - lambda_) * v_t[1:], v_t_plus_1[-1:]], axis=0
        )
    else:
        raise ValueError(f"mode {mode} is not supported")

    q_t = rewards + discounts * gae_v_t_plus_1
    gae_adv = q_t - v_t

    return gae_adv
