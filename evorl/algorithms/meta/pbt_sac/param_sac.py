import logging
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import optax

from evorl.agent import AgentState
from evorl.envs import Space, Box, create_env, AutoresetMode
from evorl.evaluators import Evaluator
from evorl.distributed import agent_gradient_update, psum
from evorl.distribution import get_tanh_norm_dist
from evorl.metrics import MetricBase
from evorl.rollout import rollout
from evorl.replay_buffers import ReplayBuffer
from evorl.types import PyTreeDict, State, LossDict
from evorl.sample_batch import SampleBatch
from evorl.networks import make_policy_network, make_q_network
from evorl.utils import running_statistics
from evorl.utils.jax_utils import scan_and_mean, tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update

from ...offpolicy_utils import clean_trajectory, OffPolicyWorkflowTemplate
from ...sac import SACTrainMetric, SACAgent


logger = logging.getLogger(__name__)


class ParamSACTrainMetric(SACTrainMetric):
    trajectory: SampleBatch = None


class ParamSACAgent(SACAgent):
    """
    SAC agent with parameterized hyperparameters and adaptive alpha.
    """

    target_entropy_h: float = 1.0

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        agent_state = super().init(obs_space, action_space, key)

        return agent_state.replace(
            extra_state=agent_state.extra_state.replace(
                discount_g=jnp.log(1 - self.discount),  # discount = 1 - exp(g)
                target_entropy_h=jnp.float32(
                    self.target_entropy_h
                ),  # target_entropy = h*(-|A|)
            )
        )

    def alpha_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        raw_actions = self.actor_network.apply(agent_state.params.actor_params, obs)
        actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
        actions = actions_dist.sample(seed=key)
        actions_logp = actions_dist.log_prob(actions)

        target_entropy = (
            agent_state.extra_state.target_entropy_h
            * agent_state.extra_state.target_entropy
        )
        # official impl:
        alpha = jnp.exp(agent_state.params.log_alpha)
        alpha_loss = jnp.mean(
            -alpha * jax.lax.stop_gradient(actions_logp + target_entropy)
        )

        # another impl: see stable-baselines3/issues/36
        # alpha_loss = (- agent_state.params.log_alpha *
        #               jax.lax.stop_gradient(actions_logp + target_entropy)).mean()

        return PyTreeDict(
            alpha_loss=alpha_loss, log_alpha=agent_state.params.log_alpha, alpha=alpha
        )

    def critic_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        obs = sample_batch.obs
        next_obs = sample_batch.extras.env_extras.ori_obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )

        discounts = (1 - jnp.exp(agent_state.extra_state.discount_g)) * (
            1 - sample_batch.extras.env_extras.termination
        )

        alpha = jnp.exp(agent_state.params.log_alpha)

        # [B, 2]
        qs = self.critic_network.apply(
            agent_state.params.critic_params, obs, sample_batch.actions
        )

        next_raw_actions = self.actor_network.apply(
            agent_state.params.actor_params, next_obs
        )
        next_actions_dist = get_tanh_norm_dist(*jnp.split(next_raw_actions, 2, axis=-1))
        next_actions = next_actions_dist.sample(seed=key)
        next_actions_logp = next_actions_dist.log_prob(next_actions)
        # [B, 2]
        next_qs = self.critic_network.apply(
            agent_state.params.target_critic_params, next_obs, next_actions
        )
        qs_target = sample_batch.rewards * self.reward_scale + discounts * (
            jnp.min(next_qs, axis=-1) - alpha * next_actions_logp
        )
        qs_target = jnp.repeat(qs_target[..., None], 2, axis=-1)

        q_loss = optax.squared_error(qs, qs_target).sum(-1).mean()
        return PyTreeDict(critic_loss=q_loss)


def make_mlp_sac_agent(
    action_space: Space,
    critic_hidden_layer_sizes: tuple[int] = (256, 256),
    actor_hidden_layer_sizes: tuple[int] = (256, 256),
    init_alpha: float = 1.0,
    discount: float = 0.99,
    reward_scale: float = 1.0,
    target_entropy_h: float = 1.0,
    normalize_obs: bool = False,
):
    if isinstance(action_space, Box):
        action_size = action_space.shape[0] * 2
    else:
        raise NotImplementedError(f"Unsupported action space: {action_space}")

    actor_network = make_policy_network(
        action_size=action_size,  # mean+std
        hidden_layer_sizes=actor_hidden_layer_sizes,
    )

    critic_network = make_q_network(
        n_stack=2,
        hidden_layer_sizes=critic_hidden_layer_sizes,
    )

    if normalize_obs:
        obs_preprocessor = running_statistics.normalize
    else:
        obs_preprocessor = None

    return ParamSACAgent(
        critic_network=critic_network,
        actor_network=actor_network,
        obs_preprocessor=obs_preprocessor,
        init_alpha=init_alpha,
        discount=discount,
        reward_scale=reward_scale,
        target_entropy_h=target_entropy_h,
    )


class ParamSACWorkflow(OffPolicyWorkflowTemplate):
    """
    This workflow can only be used with PBTParamSACWorkflow, since the replay buffer is initialized and managed by PBT.
    """

    @classmethod
    def name(cls):
        return "ParamSAC"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            env_name=config.env.env_name,
            env_type=config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )

        agent = make_mlp_sac_agent(
            action_space=env.action_space,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            init_alpha=config.alpha,
            discount=config.discount,
            reward_scale=config.reward_scale,
            target_entropy_h=config.target_entropy_h,
            normalize_obs=config.normalize_obs,
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
            min_sample_timesteps=max(
                config.batch_size, config.learning_start_timesteps
            ),
            sample_batch_size=config.batch_size,
        )

        eval_env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
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
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree]:
        agent_state = self.agent.init(self.env.obs_space, self.env.action_space, key)
        opt_state = PyTreeDict(
            dict(
                actor=self.optimizer.init(agent_state.params.actor_params),
                critic=self.optimizer.init(agent_state.params.critic_params),
                alpha=self.optimizer.init(agent_state.params.log_alpha),
            )
        )

        return agent_state, opt_state

    def setup(self, key: chex.PRNGKey) -> State:
        key, agent_key, env_key = jax.random.split(key, 3)

        agent_state, opt_state = self._setup_agent_and_optimizer(agent_key)
        workflow_metrics = self._setup_workflow_metrics()
        env_state = self.env.reset(env_key)

        state = State(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state,
            replay_buffer_state=None,
            hp_state=PyTreeDict(
                actor_loss_weight=jnp.float32(1.0),
                critic_loss_weight=jnp.float32(1.0),
                alpha_loss_weight=jnp.float32(1.0),
            ),
        )

        return state.replace()

    def step(self, state: State) -> tuple[MetricBase, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

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
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    pmap_axis_name=self.pmap_axis_name,
                )
            )

        # Here replay_buffer_state is read-only,
        # we save the data externally instead
        replay_buffer_state = state.replay_buffer_state

        # replay_buffer_state = self.replay_buffer.add(
        #     state.replay_buffer_state, trajectory
        # )

        def critic_loss_fn(agent_state, sample_batch, key):
            loss_dict = self.agent.critic_loss(agent_state, sample_batch, key)

            loss = loss_dict.critic_loss * state.hp_state.critic_loss_weight
            return loss, loss_dict

        def actor_loss_fn(agent_state, sample_batch, key):
            loss_dict = self.agent.actor_loss(agent_state, sample_batch, key)

            loss = loss_dict.actor_loss * state.hp_state.actor_loss_weight
            return loss, loss_dict

        def alpha_loss_fn(agent_state, sample_batch, key):
            loss_dict = self.agent.alpha_loss(agent_state, sample_batch, key)

            loss = loss_dict.alpha_loss * state.hp_state.alpha_loss_weight
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

        alpha_update_fn = agent_gradient_update(
            alpha_loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True,
            attach_fn=lambda agent_state, log_alpha: agent_state.replace(
                params=agent_state.params.replace(log_alpha=log_alpha)
            ),
            detach_fn=lambda agent_state: agent_state.params.log_alpha,
        )

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state = carry

            critic_opt_state = opt_state.critic
            actor_opt_state = opt_state.actor

            key, critic_key, actor_key, alpha_key, rb_key = jax.random.split(key, num=5)

            if self.config.actor_update_interval - 1 > 0:

                def _sample_and_update_critic_fn(carry, unused_t):
                    key, agent_state, critic_opt_state = carry

                    key, rb_key, critic_key = jax.random.split(key, num=3)
                    # it's safe to use read-only replay_buffer_state here.
                    sample_batch = self.replay_buffer.sample(
                        replay_buffer_state, rb_key
                    )

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

            sample_batch = self.replay_buffer.sample(replay_buffer_state, rb_key)

            (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
                critic_update_fn(
                    critic_opt_state, agent_state, sample_batch, critic_key
                )
            )

            (actor_loss, actor_loss_dict), agent_state, actor_opt_state = (
                actor_update_fn(actor_opt_state, agent_state, sample_batch, actor_key)
            )

            opt_state = opt_state.replace(
                actor=actor_opt_state, critic=critic_opt_state
            )

            # we follow the update order of the official implementation:
            # critic -> actor -> alpha
            alpha_opt_state = opt_state.alpha
            (alpha_loss, alpha_loss_dict), agent_state, alpha_opt_state = (
                alpha_update_fn(alpha_opt_state, agent_state, sample_batch, alpha_key)
            )
            opt_state = opt_state.replace(alpha=alpha_opt_state)

            alpha_loss_dict = alpha_loss_dict.replace(
                log_alpha=agent_state.params.log_alpha,
                alpha=jnp.exp(agent_state.params.log_alpha),
            )

            res = (
                critic_loss,
                actor_loss,
                alpha_loss,
                critic_loss_dict,
                actor_loss_dict,
                alpha_loss_dict,
            )

            target_critic_params = soft_target_update(
                agent_state.params.target_critic_params,
                agent_state.params.critic_params,
                self.config.tau,
            )
            agent_state = agent_state.replace(
                params=agent_state.params.replace(
                    target_critic_params=target_critic_params
                )
            )

            return (key, agent_state, opt_state), res

        (
            (_, agent_state, opt_state),
            (
                critic_loss,
                actor_loss,
                alpha_loss,
                critic_loss_dict,
                actor_loss_dict,
                alpha_loss_dict,
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
            alpha_loss=alpha_loss,
            raw_loss_dict=PyTreeDict(
                {**critic_loss_dict, **actor_loss_dict, **alpha_loss_dict}
            ),
            trajectory=trajectory,
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
            # replay_buffer_state=replay_buffer_state,
            opt_state=opt_state,
        )
