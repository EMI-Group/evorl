import jax
import jax.numpy as jnp
from flax import struct

from evorl.types import SampleBatch

from evorl.networks import make_policy_network, make_value_network
from evorl.utils import running_statistics
from evorl.utils.distribution import TanhNormal
from evorl.utils.jax_utils import tree_concat
from evorl.utils.toolkits import compute_gae
from evorl.workflows import OnPolicyRLWorkflow
from evorl.agents import AgentState
from evorl.distributed import agent_gradient_update, psum, pmean
from evorl.envs import create_env, Env
from evorl.evaluator import Evaluator
from .agent import Agent, AgentState

from evox import State as EvoXState

# from typing import Dict
import chex
import distrax
import optax
from evorl.types import (
    LossDict, Action, Params, PolicyExtraInfo, EnvState,
    Observation, TrainMetric
)
from typing import Tuple, Sequence, Optional
import dataclasses


@struct.dataclass
class A2CNetworkParams:
    """Contains training state for the learner."""
    policy_params: Params
    value_params: Params


@struct.dataclass
class A2CTrainMetric(TrainMetric):
    train_total_reward: float = 0


@struct.dataclass
class A2CAgent(Agent):
    # policy_network: nn.Module
    # value_network: nn.Module
    actor_hidden_layer_sizes: Tuple[int] = (256, 256)
    critic_hidden_layer_sizes: Tuple[int] = (256, 256)
    normalize_obs: bool = False
    continuous_action: bool = False
    gae_lambda: float = 0.95
    discount: float = 0.99

    def init(self, key: chex.PRNGKey) -> AgentState:

        obs_size = self.obs_space.shape[0]

        if self.continuous_action:
            action_size = self.action_space.shape[0]
            action_size *= 2
        else:
            action_size = self.action_space.n

        policy_key, value_key, obs_preprocessor_key = jax.random.split(key, 3)
        self.policy_network, policy_init_fn = make_policy_network(
            action_size=action_size,
            obs_size=obs_size,
            hidden_layer_sizes=self.actor_hidden_layer_sizes
        )
        policy_params = policy_init_fn(policy_key)

        self.value_network, value_init_fn = make_value_network(
            obs_size=obs_size,
            hidden_layer_sizes=self.critic_hidden_layer_sizes
        )
        value_params = value_init_fn(value_key)

        params_state = A2CNetworkParams(
            policy_params=policy_params,
            value_params=value_params
        )

        if self.normalize_obs:
            self.obs_preprocessor = running_statistics.normalize
            dummy_obs = self.obs_space.sample(obs_preprocessor_key)
            # dummy_obs = jax.broadcast_to(dummy_obs, (self.env.num_envs,) + dummy_obs.shape)
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
        else:
            obs_preprocessor_state = None

        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state
        )

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        """
            Args:
                sample_barch: [#env, ...]
        """
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(
            agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = TanhNormal(
                *jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = distrax.Categorical(logits=raw_actions)

        actions = actions_dist.sample(seed=key)

        policy_extras = dict(
            raw_action=raw_actions,
            logp=actions_dist.log_prob(actions)
        )

        return jax.lax.stop_gradient(actions), policy_extras

    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        """
            Args:
                sample_barch: [#env, ...]
        """
        obs = sample_batch.obs
        if self.normalize_obs:
            obs = self.obs_preprocessor(
                obs, agent_state.obs_preprocessor_state)

        raw_actions = self.policy_network.apply(
            agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = TanhNormal(
                *jnp.split(raw_actions, 2, axis=-1))
        else:
            actions_dist = distrax.Categorical(logits=raw_actions)

        actions = actions_dist.mode()

        return jax.lax.stop_gradient(actions), {}

    def loss(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
        """

            sample_batch: [T, #envs, ...]


            Return: LossDict[
                actor_loss
                critic_loss
                actor_entropy_loss
            ]
        """

        last_obs = sample_batch.extras['env_extras']['last_obs']
        v_obs = tree_concat(
            sample_batch.obs,
            last_obs[-1:]
        )
        if self.normalize_obs:
            v_obs = self.obs_preprocessor(
                v_obs, agent_state.obs_preprocessor_state)

        # ======= critic =======
        # concated [values, bootstrap_value]
        vs = self.compute_values(agent_state, v_obs)

        # peb_rewards = sample_batch.info["policy_extras"]["peb_rewards"]

        v_targets, advantages = compute_gae(
            rewards=sample_batch.rewards,  # peb_rewards
            values=vs,
            dones=sample_batch.dones,
            gae_lambda=self.gae_lambda,
            discount=self.discount
        )

        # ====== actor =======
        obs = v_obs[:-1]

        # [T, B, A]
        raw_actions = self.policy_network.apply(
            agent_state.params.policy_params, obs)

        if self.continuous_action:
            actions_dist = TanhNormal(
                *jnp.split(raw_actions, 2, axis=-1))
            eps = 1e-6
            # [T, B]
            actions_logp = actions_dist.log_prob(
                jnp.clip(
                    jax.lax.stop_gradient(sample_batch.actions),
                    -1+eps, 1-eps
                ))  # avoid nan logp
        else:
            actions_dist = distrax.Categorical(logits=raw_actions)
            actions_logp = actions_dist.log_prob(
                jax.lax.stop_gradient(sample_batch.actions))

        # advantages: [T, B]
        policy_loss = - (advantages * actions_logp).mean()
        value_loss = optax.huber_loss(vs[:-1], v_targets, delta=1).mean()

        if self.continuous_action:
            # TODO: check correctness
            entropy_loss = - (jnp.exp(actions_logp) * actions_logp).mean()
        else:
            entropy_loss = actions_dist.entropy().mean()

        return dict(
            actor_loss=policy_loss,
            critic_loss=value_loss,
            actor_entropy_loss=entropy_loss
        )

    def compute_values(self, agent_state: AgentState, obs: Observation) -> chex.Array:
        """
            Args:
                obs: (normalized) obs
        """
        return self.value_network.apply(
            agent_state.params.value_params, obs).squeeze(-1)


class A2CWorkflow(OnPolicyRLWorkflow):
    def __init__(self, config):
        env = create_env(
            config.env,
            config.env_type,
            episode_length=1000,
            parallel=config.num_envs,
            autoreset=True
        )

        agent = A2CAgent(
            action_space=env.action_space,
            obs_space=env.obs_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            normalize_obs=config.normalize_obs,
            continuous_action=config.agent_network.continuous_action,
            gae_lambda=config.gae_lambda,
            discount=config.discount
        )
        optimizer = optax.adam(config.optimizer.lr)

        eval_env = create_env(
            config.env,
            config.env_type,
            episode_length=1000,
            parallel=config.num_eval_envs,
            autoreset=True
        )

        evaluator = Evaluator(
            env=eval_env, agent=agent, max_episode_length=1000, discount=1.0)

        super(A2CWorkflow, self).__init__(
            config, env, agent, optimizer, evaluator)

    def step(self, state: EvoXState):

        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        # trajectory: [T, #envs, ...]
        env_state, trajectory = rollout(
            self.env,
            self.agent,
            state.env_state,
            state.agent_state,
            rollout_key,
            rollout_length=self.config.rollout_length,
            extra_fields=('last_obs',)
        )

        agent_state = state.agent_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state, trajectory.obs,
                    pmap_axis_name=self.pmap_axis_name
                )
            )

        def loss_fn(agent_state, sample_batch, key):
            # learn all data from trajectory
            loss_dict = self.agent.loss(agent_state, sample_batch, key)
            loss_weights = self.config.optimizer.loss_weights
            loss = jnp.zeros(())
            for i, loss_key in enumerate(loss_weights.keys()):
                loss += loss_weights[loss_key] * loss_dict[loss_key]

            return loss, loss_dict

        update_fn = agent_gradient_update(
            loss_fn,
            self.optimizer,
            pmap_axis_name=self.pmap_axis_name,
            has_aux=True)

        opt_state, (loss, metrics), agent_state = update_fn(
            state.opt_state,
            agent_state,
            trajectory,
            learn_key
        )

        env_timesteps = psum(
            state.metric.env_timesteps +
            self.config.rollout_length * self.config.num_envs,
            axis_name=self.pmap_axis_name
        )

        metric = TrainMetric(
            env_timesteps=env_timesteps,
            iterations=state.metric.iterations + 1
        )

        jax.debug.print("total loss:{x}", x=loss)
        jax.debug.print("losses: {x}", x=metrics)
        jax.debug.print("metric {x}", x=metric)

        return state.update(
            key=key,
            metric=metric,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state
        )

    def learn(self, state: EvoXState) -> EvoXState:
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        num_iters = self.config.total_timesteps // one_step_timesteps

        jit_evaluate = jax.jit(self.evaluator.evaluate, static_argnums=(0,))

        for i in range(num_iters, 1):
            state = self.step(state)

            if i % self.config.eval_interval == 0:
                state = self.evaluate(state)

        return state

    def evaluate(self, state: EvoXState) -> EvoXState:
        key, eval_key = jax.random.split(state.key, num=2)
        eval_metrics = self.evaluator.evaluate(
            state.agent_state,
            num_episodes=self.config.eval_episodes,
            key=eval_key
        )

        discount_return = eval_metrics.discount_return.mean()
        discount_return = pmean(
            discount_return, axis_name=self.pmap_axis_name)

        jax.debug.print("discount return: {x}", x=discount_return)

        state = state.update(key=key)
        return state

    def enable_multi_devices(self, state: EvoXState, devices: Optional[Sequence[jax.Device]] = None) -> EvoXState:
        num_devices = len(devices)
        self.config.rollout_length = self.config.rollout_length // num_devices
        self.config.eval_episodes = self.config.eval_episodes // num_devices
        return super().enable_multi_devices(state, devices)


def env_step(
    env: Env,
    agent: Agent,
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    sample_batch: SampleBatch,
    key: chex.PRNGKey,
    env_extra_fields: Sequence[str] = (),
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect data.
    """

    actions, policy_extras = agent.compute_actions(
        agent_state, sample_batch, key)
    env_nstate = env.step(env_state, actions)

    info = env_nstate.info
    env_extras = {x: info[x] for x in env_extra_fields if x in info}

    transition = SampleBatch(
        obs=env_state.obs,
        actions=actions,
        rewards=env_nstate.reward,
        dones=env_nstate.done,
        # next_obs=env_nstate.info["last_obs"],
        next_obs=env_nstate.obs,
        extras=dict(
            policy_extras=policy_extras,
            env_extras=env_extras
        ))

    return env_nstate, transition


def rollout(
    env: Env,
    agent: A2CAgent,
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    extra_fields: Sequence[str] = ('last_obs',)
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect given rollout_length trajectory.

        Args:
            env: vampped env w/ autoreset
        Returns:
            env_state: last env_state after rollout
            trajectory: SampleBatch [T, #envs, ...], T=rollout_length
    """

    def _one_step_rollout(carry, unused_t):
        """
            sample_batch: one-step obs
            transition: one-step full info
        """
        env_state, current_key = carry
        next_key, current_key = jax.random.split(current_key, 2)

        # sample_batch: [#envs, ...]
        sample_batch = SampleBatch(
            obs=env_state.obs,
        )

        # Note: will XLA optimize repeated calls?
        # transition: [#envs, ...]
        env_nstate, transition = env_step(
            env, agent, env_state, agent_state,
            sample_batch, current_key, extra_fields
        )

        # set PEB reward for GAE:
        truncation = env_nstate.info['truncation']  # [#envs]
        # Note: if truncation happens in any env in the batch, apply PEB
        rewards = transition.rewards + agent.discount * jax.lax.cond(
            truncation.any(),
            lambda last_obs: agent.compute_values(
                agent_state, last_obs) * truncation,
            lambda last_obs: jnp.zeros_like(transition.rewards),
            env_nstate.info["last_obs"]  # [#envs, ...]
        )

        transition = transition.replace(rewards=rewards)
        # transition.info["policy_extras"]["peb_rewards"] = reward # ok for dict

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key), (), length=rollout_length)

    return env_state, trajectory
