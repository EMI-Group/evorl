"""Continuous MADDPG with local actors and independent centralized critics."""

import math

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from evorl.agent import AgentState
from evorl.distributed import psum, pmean_nnx_loss_and_grad
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators import MultiAgentEvaluator
from evorl.metrics import TrainMetric
from evorl.networks import make_policy_network, make_q_network
from evorl.replay_buffers import ReplayBuffer
from evorl.rollout import multi_agent_rollout
from evorl.types import PyTreeDict, pytree_field
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update

from .common import (
    MultiAgentAgent,
    agent_layout,
    centralized_features,
    init_preprocessing,
    local_features,
    masked_mean,
    scale_action,
    update_preprocessing,
)
from .offpolicy import MultiAgentOffPolicyWorkflow, joint_actions


class MADDPGAgent(MultiAgentAgent):
    """One bounded deterministic actor and one centralized critic per agent."""

    num_critics: int = pytree_field(default=1, static=True)
    discount: float = 0.99
    exploration_noise: float = 0.1

    def init(self, obs_space, action_space, key):
        if not all(self.continuous_actions):
            raise TypeError(
                "MADDPG/MATD3 currently require continuous Box actions for every agent."
            )
        models = nnx.Dict()
        keys = jax.random.split(key, 2 * len(self.agents))
        local_sizes = [
            sum(map(math.prod, shapes)) for shapes in self.observation_shapes
        ]
        central_size = (
            sum(map(math.prod, self.global_observation_shapes))
            if self.global_observation_shapes is not None
            else sum(local_sizes)
        )
        cfg = self.agent_network_cfg
        for index, name in enumerate(self.agents):
            actor = make_policy_network(
                input_size=local_sizes[index],
                action_size=self.action_sizes[index],
                hidden_layer_sizes=cfg.actor_hidden_layer_sizes,
                activation_final=jnp.tanh,
                norm_layer_type=cfg.get("norm_layer_type", "none"),
                rngs=nnx.Rngs(keys[2 * index]),
            )
            critic = make_q_network(
                input_size=central_size,
                action_size=sum(self.action_sizes),
                n_stack=self.num_critics,
                hidden_layer_sizes=cfg.critic_hidden_layer_sizes,
                norm_layer_type=cfg.get("norm_layer_type", "none"),
                rngs=nnx.Rngs(keys[2 * index + 1]),
            )
            models[name] = nnx.Dict(
                actor=actor,
                critic=critic,
                target_actor=nnx.clone(actor),
                target_critic=nnx.clone(critic),
            )
        return AgentState(
            model_state=nnx.unpack(models),
            obs_preprocessor_state=init_preprocessing(self),
        )

    def compute_actions(self, agent_state, sample_batch, key):
        models = self.get_models(agent_state, copy=True)
        features = local_features(self, agent_state, sample_batch.obs)
        keys = jax.random.split(key, len(self.agents))
        actions = PyTreeDict()
        for index, name in enumerate(self.agents):
            normalized = models[name].actor(features[name])
            noise = (
                jax.random.normal(keys[index], normalized.shape)
                * self.exploration_noise
            )
            actions[name] = scale_action(
                jnp.clip(normalized + noise, -1, 1),
                self.action_bounds[index],
                self.action_shapes[index],
            )
        return actions, PyTreeDict()

    def evaluate_actions(self, agent_state, sample_batch, key):
        models = self.get_models(agent_state, copy=True)
        features = local_features(self, agent_state, sample_batch.obs)
        return PyTreeDict(
            {
                name: scale_action(
                    models[name].actor(features[name]),
                    self.action_bounds[index],
                    self.action_shapes[index],
                )
                for index, name in enumerate(self.agents)
            }
        ), PyTreeDict()

    def critic_loss(self, agent_state, sample_batch, key, agent_id, axis_name=None):
        models = self.get_models(agent_state)
        features = local_features(self, agent_state, sample_batch.next_obs)
        next_actions = PyTreeDict(
            {
                name: scale_action(
                    models[name].target_actor(features[name]),
                    self.action_bounds[index],
                    self.action_shapes[index],
                )
                for index, name in enumerate(self.agents)
            }
        )
        next_state = centralized_features(
            self,
            agent_state,
            sample_batch.next_obs,
            sample_batch.extras.next_global_obs,
        )
        next_q = models[agent_id].target_critic(
            next_state, joint_actions(self, next_actions)
        )
        target = jax.lax.stop_gradient(
            sample_batch.rewards[agent_id]
            + self.discount * (1 - sample_batch.extras.terminations[agent_id]) * next_q
        )
        inputs = centralized_features(
            self, agent_state, sample_batch.obs, sample_batch.extras.global_obs
        )
        q = models[agent_id].critic(inputs, joint_actions(self, sample_batch.actions))
        mask = sample_batch.extras.active[agent_id]
        return PyTreeDict(
            critic_loss=masked_mean(optax.squared_error(q, target), mask, axis_name),
            q_value=masked_mean(q, mask, axis_name),
        )

    def actor_loss(self, agent_state, sample_batch, key, agent_id, axis_name=None):
        models = self.get_models(agent_state)
        features = local_features(self, agent_state, sample_batch.obs)
        index = self.agents.index(agent_id)
        # Other agents' actions come from this same joint replay transition.
        actions = sample_batch.actions.replace(
            **{
                agent_id: scale_action(
                    models[agent_id].actor(features[agent_id]),
                    self.action_bounds[index],
                    self.action_shapes[index],
                )
            }
        )
        inputs = centralized_features(
            self, agent_state, sample_batch.obs, sample_batch.extras.global_obs
        )
        q = models[agent_id].critic(inputs, joint_actions(self, actions))
        return PyTreeDict(
            actor_loss=-masked_mean(q, sample_batch.extras.active[agent_id], axis_name)
        )


class MADDPGWorkflow(MultiAgentOffPolicyWorkflow):
    """Train each decentralized actor against its own centralized joint critic."""

    agent_class = MADDPGAgent

    @classmethod
    def name(cls):
        return "MADDPG"

    @classmethod
    def _build_from_config(cls, config):
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )
        layout = agent_layout(env)
        if not all(layout["continuous_actions"]):
            raise TypeError(
                "MADDPG/MATD3 require continuous Box actions; configure the task's continuous action mode."
            )
        if (
            min(
                config.rollout_length,
                config.num_updates_per_iter,
                config.batch_size,
                config.replay_buffer_capacity,
            )
            <= 0
        ):
            raise ValueError("Rollout, update, batch and buffer sizes must be positive")
        if config.replay_buffer_capacity < max(
            config.batch_size, config.rollout_length * config.num_envs
        ):
            raise ValueError("Replay capacity must fit a rollout and a training batch")
        if (
            min(
                config.random_timesteps,
                config.learning_start_timesteps,
                config.total_timesteps,
            )
            < 0
        ):
            raise ValueError("Sampling budgets must be nonnegative")
        warmup_size = max(
            config.random_timesteps // config.num_envs * config.num_envs,
            config.learning_start_timesteps,
        )
        if warmup_size < config.batch_size:
            raise ValueError(
                "Warmup must provide at least batch_size joint transitions"
            )
        agent = cls.agent_class(
            **layout,
            agent_network_cfg=config.agent_network,
            discount=config.discount,
            exploration_noise=config.exploration_noise,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
            **{
                name: config[name]
                for name in ("policy_noise", "clip_policy_noise")
                if name in config
            },
        )
        optimizer = optax.adam(config.optimizer.lr)
        if config.optimizer.grad_clip_norm:
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.optimizer.grad_clip_norm), optimizer
            )
        buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            sample_batch_size=config.batch_size,
            min_sample_timesteps=config.batch_size,
        )
        eval_env = create_env(
            config.env,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )
        evaluator = MultiAgentEvaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )
        return cls(env, agent, optimizer, evaluator, buffer, config)

    def step(self, state):
        key, rollout_key, learn_key = jax.random.split(state.key, 3)
        if self.is_multi_device:
            rank = jax.lax.axis_index(self.dp_axis_name)
            rollout_key, learn_key = (
                jax.random.fold_in(k, rank) for k in (rollout_key, learn_key)
            )
        trajectory, env_state = multi_agent_rollout(
            self.env,
            self.agent.compute_actions,
            state.env_state,
            state.agent_state,
            rollout_key,
            self.config.rollout_length,
        )
        episode_done = trajectory.extras.episode_done
        trajectory = tree_stop_gradient(flatten_rollout_trajectory(trajectory))
        agent_state = update_preprocessing(
            self.agent, state.agent_state, trajectory, self.dp_axis_name
        )
        buffer_state = self.replay_buffer.add(state.replay_buffer_state, trajectory)

        def update(carry, _):
            agent_state, opt_state, key = carry
            key, sample_key, critic_key, actor_key = jax.random.split(key, 4)
            batch = self.replay_buffer.sample(buffer_state, sample_key)
            models = nnx.merge(agent_state.model_state, copy=True)
            losses = PyTreeDict()
            total_loss = jnp.float32(0)
            for name in self.agent.agents:

                def critic_loss(models, name=name):
                    info = self.agent.critic_loss(
                        agent_state.replace(model_state=nnx.unpack(models)),
                        batch,
                        critic_key,
                        name,
                        self.dp_axis_name,
                    )
                    return info.critic_loss, info

                (value, info), grads = pmean_nnx_loss_and_grad(
                    critic_loss,
                    self.dp_axis_name,
                    has_aux=True,
                    wrt=lambda path, variable, name=name: (
                        path[:2] == (name, "critic") and isinstance(variable, nnx.Param)
                    ),
                )(models)
                optimizer = nnx.merge(opt_state[name].critic, copy=True)
                optimizer.update(models[name].critic, grads[name]["critic"])
                opt_state = opt_state.replace(
                    **{name: opt_state[name].replace(critic=nnx.unpack(optimizer))}
                )
                losses[name] = info
                total_loss += value
            for name in self.agent.agents:
                critic_state = nnx.clone(
                    nnx.state(models[name].critic, nnx.Not(nnx.Param))
                )

                def actor_loss(models, name=name):
                    return self.agent.actor_loss(
                        agent_state.replace(model_state=nnx.unpack(models)),
                        batch,
                        actor_key,
                        name,
                        self.dp_axis_name,
                    ).actor_loss

                value, grads = pmean_nnx_loss_and_grad(
                    actor_loss,
                    self.dp_axis_name,
                    wrt=lambda path, variable, name=name: (
                        path[:2] == (name, "actor") and isinstance(variable, nnx.Param)
                    ),
                )(models)
                optimizer = nnx.merge(opt_state[name].actor, copy=True)
                optimizer.update(models[name].actor, grads[name]["actor"])
                opt_state = opt_state.replace(
                    **{name: opt_state[name].replace(actor=nnx.unpack(optimizer))}
                )
                nnx.update(models[name].critic, critic_state)
                losses[name] = losses[name].replace(actor_loss=value)
                total_loss += value
                for target, online in (
                    (models[name].target_actor, models[name].actor),
                    (models[name].target_critic, models[name].critic),
                ):
                    nnx.update(
                        target,
                        soft_target_update(
                            nnx.state(target, nnx.Param),
                            nnx.state(online, nnx.Param),
                            self.config.tau,
                        ),
                    )
            return (
                agent_state.replace(model_state=nnx.unpack(models)),
                opt_state,
                key,
            ), (total_loss, losses)

        (agent_state, opt_state, _), (loss, losses) = jax.lax.scan(
            update,
            (agent_state, state.opt_state, learn_key),
            (),
            length=self.config.num_updates_per_iter,
        )
        losses = jax.tree.map(lambda x: x.mean(axis=0), losses)
        metrics = state.metrics.replace(
            iterations=state.metrics.iterations + 1,
            sampled_timesteps=state.metrics.sampled_timesteps
            + psum(
                jnp.uint32(self.config.rollout_length * self.config.num_envs),
                self.dp_axis_name,
            ),
            sampled_episodes=state.metrics.sampled_episodes
            + psum(episode_done.sum().astype(jnp.uint32), self.dp_axis_name),
        )
        train_metrics = TrainMetric(
            loss=loss.mean(),
            raw_loss_dict=losses,
        ).all_reduce(self.dp_axis_name)
        return train_metrics, state.replace(
            key=key,
            env_state=env_state,
            agent_state=agent_state,
            opt_state=opt_state,
            metrics=metrics,
            replay_buffer_state=buffer_state,
        )
