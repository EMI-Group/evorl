from evorl.utils import running_statistics
import logging
from flax import nnx
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import optax

from evorl.agent import AgentState
from evorl.envs import Space, create_env, AutoresetMode
from evorl.evaluators import Evaluator
from evorl.distributed import pmean_nnx_loss_and_grad, psum
from evorl.distribution import get_tanh_norm_dist
from evorl.metrics import MetricBase
from evorl.rollout import rollout
from evorl.replay_buffers import ReplayBuffer
from evorl.types import PyTreeDict, State, LossDict
from evorl.sample_batch import SampleBatch
from evorl.utils.jax_utils import scan_and_mean, tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update

from evorl.algorithms.offpolicy_utils import clean_trajectory, OffPolicyWorkflowTemplate
from evorl.algorithms.sac import SACTrainMetric, SACAgent


logger = logging.getLogger(__name__)


class ParamSACTrainMetric(SACTrainMetric):
    trajectory: SampleBatch = None


class ParamSACAgent(SACAgent):
    """SAC agent with parameterized hyperparameters."""

    def init(self, obs_space: Space, action_space: Space, key: jax.Array) -> AgentState:
        agent_state = super().init(obs_space, action_space, key)

        return agent_state.replace(
            extra_state=agent_state.extra_state.replace(
                discount_g=-jnp.log(1 - jnp.float32(self.discount)),
            )
        )

    def critic_loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: jax.Array,
    ) -> LossDict:
        models = self.get_models(agent_state)
        obs = sample_batch.obs
        next_obs = sample_batch.extras.env_extras.ori_obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )

        discounts = (1 - jnp.exp(-agent_state.extra_state.discount_g)) * (
            1 - sample_batch.extras.env_extras.termination
        )

        alpha = jnp.exp(models.log_alpha[...])

        # [B, 2]
        qs = models.critic(obs, sample_batch.actions)

        next_raw_actions = models.actor(next_obs)
        next_actions_dist = get_tanh_norm_dist(*jnp.split(next_raw_actions, 2, axis=-1))
        next_actions = next_actions_dist.sample(seed=key)
        next_actions_logp = next_actions_dist.log_prob(next_actions)
        # [B, 2]
        next_qs = models.target_critic(next_obs, next_actions)
        qs_target = sample_batch.rewards + discounts * (
            jnp.min(next_qs, axis=-1) - alpha * next_actions_logp
        )
        qs_target = jax.lax.stop_gradient(
            jnp.broadcast_to(qs_target[..., None], qs.shape)
        )

        q_loss = optax.squared_error(qs, qs_target).sum(-1).mean()
        return PyTreeDict(critic_loss=q_loss, q_value=qs.mean())


class ParamSACWorkflow(OffPolicyWorkflowTemplate):
    """Workflow for ParamSAC.

    Note: This workflow can only work with PBTParamSACWorkflow, since the replay buffer is initialized and managed by PBT externally.
    """

    @classmethod
    def name(cls):
        return "ParamSAC"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )

        agent = ParamSACAgent(
            agent_network_cfg=config.agent_network,
            init_alpha=config.alpha,
            discount=config.discount,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
        )

        # TODO: use different lr for critic and actor
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

        replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            min_sample_timesteps=config.batch_size,
            sample_batch_size=config.batch_size,
        )

        eval_env = create_env(
            config.env,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        return cls(
            env,
            agent,
            optimizer,
            evaluator,
            replay_buffer,
            config,
        )

    def _setup_agent_and_optimizer(
        self, key: jax.Array
    ) -> tuple[AgentState, chex.ArrayTree]:
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        models = nnx.merge(agent_state.model_state, copy=True)
        opt_state = PyTreeDict(
            actor=nnx.unpack(
                nnx.Optimizer(models.actor, self.optimizer, wrt=nnx.Param)
            ),
            critic=nnx.unpack(
                nnx.Optimizer(models.critic, self.optimizer, wrt=nnx.Param)
            ),
        )

        return agent_state, opt_state

    def setup(self, key: jax.Array) -> State:
        key, agent_key, env_key = jax.random.split(key, 3)

        agent_state, opt_state = self._setup_agent_and_optimizer(agent_key)
        workflow_metrics = self._setup_workflow_metrics()
        env_state = self.env.reset(env_key)

        return State(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state,
            replay_buffer_state=None,  # init externally
            hp_state=PyTreeDict(
                actor_loss_weight=jnp.float32(1.0),
                critic_loss_weight=jnp.float32(1.0),
            ),
        )

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)
        if self.is_multi_device:
            rank = jax.lax.axis_index(self.dp_axis_name)
            rollout_key = jax.random.fold_in(rollout_key, rank)
            learn_key = jax.random.fold_in(learn_key, rank)

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

        trajectory_dones = trajectory.dones
        trajectory = clean_trajectory(trajectory)
        trajectory = flatten_rollout_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        agent_state = state.agent_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    axis_name=self.dp_axis_name,
                )
            )

        # Here replay_buffer_state is read-only,
        # we save the data externally instead
        replay_buffer_state = state.replay_buffer_state

        def critic_loss_fn(models, agent_state, sample_batch, key):
            loss_dict = self.agent.critic_loss(
                agent_state.replace(model_state=nnx.unpack(models)), sample_batch, key
            )

            loss = loss_dict.critic_loss * state.hp_state.critic_loss_weight
            return loss, loss_dict

        def actor_loss_fn(models, agent_state, sample_batch, key):
            loss_dict = self.agent.actor_loss(
                agent_state.replace(model_state=nnx.unpack(models)), sample_batch, key
            )

            loss = loss_dict.actor_loss * state.hp_state.actor_loss_weight
            return loss, loss_dict

        critic_loss_and_grad = pmean_nnx_loss_and_grad(
            critic_loss_fn,
            self.dp_axis_name,
            has_aux=True,
            wrt=nnx.All(nnx.Param, nnx.PathContains("critic")),
        )
        actor_loss_and_grad = pmean_nnx_loss_and_grad(
            actor_loss_fn,
            self.dp_axis_name,
            has_aux=True,
            wrt=nnx.All(nnx.Param, nnx.PathContains("actor")),
        )

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
                    )

                    models = nnx.merge(agent_state.model_state, copy=True)
                    optimizer = nnx.merge(critic_opt_state, copy=True)
                    _, grads = critic_loss_and_grad(
                        models, agent_state, sample_batch, critic_key
                    )
                    optimizer.update(models.critic, grads["critic"])
                    agent_state = agent_state.replace(model_state=nnx.unpack(models))
                    critic_opt_state = nnx.unpack(optimizer)

                    return (key, agent_state, critic_opt_state), None

                key, critic_multiple_update_key = jax.random.split(key)

                (_, agent_state, critic_opt_state), _ = jax.lax.scan(
                    _sample_and_update_critic_fn,
                    (critic_multiple_update_key, agent_state, critic_opt_state),
                    (),
                    length=self.config.actor_update_interval - 1,
                )

            sample_batch = self.replay_buffer.sample(replay_buffer_state, rb_key)

            models = nnx.merge(agent_state.model_state, copy=True)
            critic_optimizer = nnx.merge(critic_opt_state, copy=True)
            (critic_loss, critic_loss_dict), critic_grads = critic_loss_and_grad(
                models, agent_state, sample_batch, critic_key
            )
            critic_optimizer.update(models.critic, critic_grads["critic"])
            critic_opt_state = nnx.unpack(critic_optimizer)

            actor_optimizer = nnx.merge(actor_opt_state, copy=True)
            (actor_loss, actor_loss_dict), actor_grads = actor_loss_and_grad(
                models, agent_state, sample_batch, actor_key
            )
            actor_optimizer.update(models.actor, actor_grads["actor"])
            actor_opt_state = nnx.unpack(actor_optimizer)
            opt_state = opt_state.replace(
                actor=actor_opt_state, critic=critic_opt_state
            )

            res = (
                critic_loss,
                actor_loss,
                critic_loss_dict,
                actor_loss_dict,
            )

            nnx.update(
                models.target_critic,
                soft_target_update(
                    nnx.state(models.target_critic, nnx.Param),
                    nnx.state(models.critic, nnx.Param),
                    self.config.tau,
                ),
            )
            agent_state = agent_state.replace(model_state=nnx.unpack(models))

            return (key, agent_state, opt_state), res

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

        train_metrics = ParamSACTrainMetric(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
            trajectory=trajectory,
        ).all_reduce(axis_name=self.dp_axis_name)

        # calculate the number of timestep
        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.dp_axis_name,
        )
        sampled_epsiodes = psum(
            trajectory_dones.sum().astype(jnp.uint32), axis_name=self.dp_axis_name
        )

        # iterations is the number of updates of the agent
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_epsiodes,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(axis_name=self.dp_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            # replay_buffer_state=replay_buffer_state,
            opt_state=opt_state,
        )
