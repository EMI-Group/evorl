import math

import jax
import jax.numpy as jnp
from flax import struct
import flax.linen as nn

from .agent import Agent, AgentState
from evorl.networks import make_q_network, MLP
from evorl.workflows import OffPolicyRLWorkflow
from evorl.rollout import rollout, env_step
from evorl.envs import create_env, Discrete, Env, EnvState
from evorl.sample_batch import SampleBatch
from evorl.distributed import PMAP_AXIS_NAME, split_key_to_devices, tree_unpmap, agent_gradient_update, psum
from evorl.distributed.gradients import loss_and_pgrad
from evorl.evaluator import Evaluator
from evorl.types import (
    LossDict, Action, Params, PolicyExtraInfo, PyTreeDict, pytree_field, MISSING_REWARD
)
from evox import State

from omegaconf import DictConfig
from typing import Any, List, Optional, Sequence, Tuple, Callable, Dict
import orbax.checkpoint as ocp
import optax
import chex
import distrax
import dataclasses

import flashbax



import logging

from ..metrics import TrainMetric, WorkflowMetric
from ..utils import running_statistics
from ..utils.toolkits import average_episode_discount_return

logger = logging.getLogger(__name__)
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]


@struct.dataclass
class DQNNetworkParams:
    """Contains training state for the learner."""
    q_params: Params
    target_q_params: Params



class DQNAgent(Agent):
    """
        Double-DQN
    """
    q_hidden_layer_sizes: Tuple[int] = (256, 256)
    discount: float = 0.99
    exploration_epsilon: float = 0.1
    q_network: nn.Module = pytree_field(lazy_init=True)
    target_q_network: nn.Module = pytree_field(lazy_init=True)

    def init(self, key: chex.PRNGKey) -> AgentState:
        obs_size = self.obs_space.shape[0]
        action_size = self.action_space.n

        q_network, q_init_fn = make_Qnetwork(
            obs_size=obs_size,
            action_size=action_size,
            hidden_layer_sizes=self.q_hidden_layer_sizes,
        )
        self.set_frozen_attr('q_network', q_network)

        key, q_key = jax.random.split(key)

        q_params = q_init_fn(q_key)

        target_q_params = q_params

        params_states = DQNNetworkParams(
            q_params=q_params,
            target_q_params=target_q_params
        )

        return AgentState(
            params=params_states
        )

    def compute_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        """
            Args:
                sample_barch: [#env, ...]
        """
        obs = sample_batch.obs

        qs = self.q_network.apply(
            agent_state.params.q_params, obs
        )
        # TODO: use tfp.Distribution
        actions_dist = distrax.EpsilonGreedy(
            qs, epsilon=self.exploration_epsilon)
        actions = actions_dist.sample(seed=key)

        return actions, PyTreeDict(
            q_values=qs
        )

    def evaluate_actions(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> Tuple[Action, PolicyExtraInfo]:
        """
            Args:
                sample_barch: [#env, ...]
        """
        qs = self.q_network.apply(
            agent_state.params.q_params, sample_batch.obs)

        actions_dist = distrax.EpsilonGreedy(
            qs, epsilon=self.exploration_epsilon)
        actions = actions_dist.mode()

        return actions, PyTreeDict()

    def loss(self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey) -> LossDict:
        """
            Args:
                sample_batch: [B, ...]
        """
        states = sample_batch.obs
        actions = sample_batch.actions
        rewards = sample_batch.rewards
        next_states = sample_batch.next_obs
        dones = sample_batch.dones

        q_values = self.q_network.apply(agent_state.params.q_params, states)
        next_actions = jnp.argmax(q_values, axis=1)
        q_values = jnp.take_along_axis(q_values, actions[:, None], axis=1).squeeze(-1)
        

        # Double DQN_target
        next_q_values = self.q_network.apply(agent_state.params.target_q_params, next_states)
        next_q_values = jnp.take_along_axis(next_q_values, next_actions[:, None], axis=1).squeeze(-1)

        # Future rewards are not considered for the completed status
        next_q_values = next_q_values * (1 - dones)
        target_q_values = rewards + self.discount * next_q_values

        td_error = jax.lax.stop_gradient(target_q_values) - q_values
        loss = jnp.mean(jnp.square(td_error))

        return PyTreeDict(
            q_loss=loss)

    def update_target_network(self, agent_state: AgentState) -> AgentState:
        return agent_state.replace(
            params=agent_state.params.replace(
                target_q_params=agent_state.params.q_params
            )
        )



class DQNWorkflow(OffPolicyRLWorkflow):
    @staticmethod
    def _rescale_config(config, devices) -> None:
        num_devices = len(devices)

        #TODO: impl it

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=1000,
            parallel=config.num_envs,
            autoreset=True
        )


        assert isinstance(env.action_space, Discrete), "Only Discrete action space is supported."

        agent = DQNAgent(
            action_space=env.action_space,
            obs_space=env.obs_space,
            q_hidden_layer_sizes=config.agent_network.q_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon
        )

        optimizer = optax.adam(config.optimizer.lr)

        replay_buffer = flashbax.make_flat_buffer(
            max_length=config.replay_buffer.capacity,
            min_length=config.replay_buffer.min_size,
            sample_batch_size=config.train_batch_size,
            add_batch_size=config.num_envs*config.rollout_length
        )

        def _replay_buffer_init_fn(replay_buffer, key):
            # dummy_action = jnp.tile(env.action_space.sample(),
            dummy_action = env.action_space.sample(key)
            dummy_obs = env.obs_space.sample(key)

            # TODO: handle RewardDict
            dummy_reward = jnp.zeros(())
            dummy_done = jnp.zeros(())

            dummy_nest_obs = dummy_obs

            # Customize your algorithm's stored data
            dummy_sample_batch = SampleBatch(
                obs=dummy_obs,
                actions=dummy_action,
                rewards=dummy_reward,
                next_obs=dummy_nest_obs,
                dones=dummy_done,
                extras=PyTreeDict(
                    policy_extras=PyTreeDict({'q_values': jnp.zeros(env.action_space.n)}),
                    env_extras=PyTreeDict({'last_obs': dummy_obs,
                                           'episode_return': dummy_reward})
                )
            )

            replay_buffer_state = replay_buffer.init(dummy_sample_batch)

            return replay_buffer_state


        eval_env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=1000,
            parallel=config.num_envs,
            autoreset=False
        )

        evaluator = Evaluator(
            env=eval_env, agent=agent, max_episode_steps=1000)

        return cls(env, agent, optimizer, evaluator, replay_buffer, _replay_buffer_init_fn, config)


    def step(self, state: State) -> Tuple[TrainMetric, State]:

        key, rollout_key, learn_key, buffer_key = jax.random.split(state.key, num=4)

        def fill_replay_buffer(state: State):
            replay_buffer_state = state.replay_buffer_state
            env_state = state.env_state
            sample_batch = SampleBatch(
                obs=env_state.obs
            )
            actions, _ = self.agent.compute_actions(state.agent_state, sample_batch, key)
            env_nstate = self.env.step(env_state, actions)
            trajectory = SampleBatch(
                obs=env_state.obs,
                actions=actions,
                rewards=env_nstate.reward,
                next_obs=env_nstate.obs,
                dones=env_nstate.done,
                extras=PyTreeDict(
                    policy_extras=PyTreeDict({'q_values': env_nstate.info.episode_return}),
                    env_extras=PyTreeDict({'last_obs': env_nstate.info.last_obs,
                                           'episode_return': env_nstate.info.episode_return})
                )
            )
            replay_buffer_state = self.replay_buffer.add(replay_buffer_state, trajectory)
            env_state = env_nstate
            state = state.update(
                env_state=env_state,
                replay_buffer_state=replay_buffer_state
            )

            train_episode_return = average_episode_discount_return(
                env_state.info.episode_return,
                trajectory.dones,
                pmap_axis_name=self.pmap_axis_name
            ).mean()

            loss = jnp.zeros(())
            loss_dict = PyTreeDict(
                q_loss=loss
            )
            train_metrics = TrainMetric(
                train_episode_return=train_episode_return,
                loss=loss,
                raw_loss_dict=loss_dict
            ).all_reduce(pmap_axis_name=self.pmap_axis_name)

            return train_metrics, state
        
        def normal_step(state: State):
            # trajectory: [T, #envs, ...]
            env_state, trajectory = rollout(
                self.env,
                self.agent,
                state.env_state,
                state.agent_state,
                rollout_key,
                rollout_length=self.config.rollout_length,
                env_extra_fields=('last_obs', 'episode_return')
            )
            trajectory = jax.tree_util.tree_map(lambda x: jax.lax.collapse(x,0,2), trajectory)
            replay_buffer_state = self.replay_buffer.add(
                state.replay_buffer_state, trajectory
            )
            agent_state = state.agent_state
            sample_batch = self.replay_buffer.sample(replay_buffer_state, buffer_key)

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
                pmap_axis_name=self.pmap_axis_name
            )


            def loss_fn(agent_state, sample_batch, key):
                # learn all data from trajectory
                loss_dict = self.agent.loss(agent_state, sample_batch, key)
                loss_weights = self.config.optimizer.loss_weights
                loss = jnp.zeros(())
                for loss_key in loss_weights.keys():
                    loss += loss_weights[loss_key] * loss_dict[loss_key]

                return loss, loss_dict

            update_fn = agent_params_gradient_update(
                loss_fn,
                self.optimizer,
                pmap_axis_name=self.pmap_axis_name,
                has_aux=True)

            (loss, loss_dict), opt_state, agent_state = update_fn(
                state.opt_state,
                agent_state,
                sample_batch.experience.first,
                learn_key
            )
        
            # ======== update metrics ========
            train_metrics = TrainMetric(
                train_episode_return=train_episode_return,
                loss=loss,
                raw_loss_dict=loss_dict
            ).all_reduce(pmap_axis_name=self.pmap_axis_name)

            return train_metrics, state.replace(
                key=key,
                env_state=env_state,
                agent_state=agent_state,
                opt_state=opt_state,
                replay_buffer_state=replay_buffer_state
            )
        
        condition = jax.lax.lt(state.metrics.iterations, int(self.config.learning_starts/self.config.num_envs/self.config.rollout_length))
        train_metrics, state = jax.lax.cond(
            condition,
            fill_replay_buffer,
            normal_step,
            state
        )
        sampled_timesteps = psum(self.config.rollout_length * self.config.num_envs,
                                    axis_name=self.pmap_axis_name)
        
        workflow_metrics = WorkflowMetric(
                sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
                iterations=state.metrics.iterations + 1,
            ).all_reduce(pmap_axis_name=self.pmap_axis_name)
        
        return train_metrics, state.update(
                metrics=workflow_metrics
            )

    def learn(self, state: State) -> State:
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        num_iters = math.ceil(self.config.total_timesteps / one_step_timesteps)

        start_iteration = tree_unpmap(
            state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            train_metrics = tree_unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = tree_unpmap(
                workflow_metrics, self.pmap_axis_name)

            self.recorder.write(workflow_metrics.to_local_dict(), i)
            train_metric_data = train_metrics.to_local_dict()
            if train_metrics.train_episode_return==MISSING_REWARD:
                del train_metric_data['train_episode_return']
            self.recorder.write(train_metric_data, i)

            if (i+1) % self.config.target_network_update_interval == 0:
                agent_state = self.agent.update_target_network(state.agent_state)
                state = state.update(agent_state=agent_state)

            if (i+1) % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)
                eval_metrics = tree_unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write({'eval': eval_metrics.to_local_dict()}, i)
                logger.debug(eval_metrics)

            self.checkpoint_manager.save(
                i,
                args=ocp.args.StandardSave(
                    tree_unpmap(state, self.pmap_axis_name))
            )

        return state

def rollout(
    env: Env,
    agent: DQNAgent,
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    env_extra_fields: Sequence[str] = ('last_obs',),
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

        # transition: [#envs, ...]
        env_nstate, transition = env_step(
            env, agent, env_state, agent_state,
            sample_batch, current_key, env_extra_fields
        )

        return (env_nstate, next_key), transition

    # trajectory: [T, #envs, ...]
    (env_state, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key), (), length=rollout_length)

    return env_state, trajectory


def make_Qnetwork(
    obs_size: int,
    action_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu) -> nn.Module:

    Qnetwork = MLP(
        layer_sizes=list(hidden_layer_sizes) + [action_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        )
    init_fn = lambda rng: Qnetwork.init(rng, jnp.ones((1,obs_size)))

    return Qnetwork, init_fn

def agent_params_gradient_update(loss_fn: Callable[..., float],
                          optimizer: optax.GradientTransformation,
                          pmap_axis_name: Optional[str],
                          has_aux: bool = False):
    def _loss_fn(params, agent_state, sample_batch, key):
        return loss_fn(agent_state.replace(
                params=agent_state.params.replace(
                    q_params=params.q_params
                )
            ),sample_batch, key)

    loss_and_pgrad_fn = loss_and_pgrad(
        _loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

    def f(opt_state, agent_state, *args, **kwargs):
        value, grads = loss_and_pgrad_fn(
            agent_state.params, agent_state, *args, **kwargs)

        params_update, opt_state = optimizer.update(
            grads, opt_state)
        params = optax.apply_updates(agent_state.params, params_update)

        agent_state = agent_state.replace(
                params=agent_state.params.replace(
                    q_params=params.q_params
                )
            )
        return value, opt_state, agent_state

    return f