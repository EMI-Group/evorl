"""MATD3 with twin centralized critics and delayed decentralized actor updates."""

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from evorl.distributed import psum, pmean_nnx_loss_and_grad
from evorl.metrics import TrainMetric
from evorl.rollout import multi_agent_rollout
from evorl.types import PyTreeDict, pytree_field
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update

from .common import (
    centralized_features,
    local_features,
    masked_mean,
    scale_action,
    update_preprocessing,
)
from .maddpg import MADDPGAgent, MADDPGWorkflow
from .offpolicy import joint_actions


class MATD3Agent(MADDPGAgent):
    """Two independent joint critics per agent, using their minimum as target."""

    num_critics: int = pytree_field(default=2, static=True)
    policy_noise: float = 0.2
    clip_policy_noise: float = 0.5

    def critic_loss(self, agent_state, sample_batch, key, agent_id, axis_name=None):
        models = self.get_models(agent_state)
        features = local_features(self, agent_state, sample_batch.next_obs)
        keys = jax.random.split(key, len(self.agents))
        next_actions = PyTreeDict()
        for index, name in enumerate(self.agents):
            normalized = models[name].target_actor(features[name])
            noise = jnp.clip(
                jax.random.normal(keys[index], normalized.shape) * self.policy_noise,
                -self.clip_policy_noise,
                self.clip_policy_noise,
            )
            next_actions[name] = scale_action(
                jnp.clip(normalized + noise, -1, 1),
                self.action_bounds[index],
                self.action_shapes[index],
            )
        next_state = centralized_features(
            self,
            agent_state,
            sample_batch.next_obs,
            sample_batch.extras.next_global_obs,
        )
        next_q = (
            models[agent_id]
            .target_critic(next_state, joint_actions(self, next_actions))
            .min(axis=-1)
        )
        target = jax.lax.stop_gradient(
            sample_batch.rewards[agent_id]
            + self.discount * (1 - sample_batch.extras.terminations[agent_id]) * next_q
        )
        inputs = centralized_features(
            self, agent_state, sample_batch.obs, sample_batch.extras.global_obs
        )
        qs = models[agent_id].critic(inputs, joint_actions(self, sample_batch.actions))
        mask = sample_batch.extras.active[agent_id]
        return PyTreeDict(
            critic_loss=masked_mean(
                optax.squared_error(
                    qs, jnp.broadcast_to(target[..., None], qs.shape)
                ).sum(axis=-1),
                mask,
                axis_name,
            ),
            q_value=masked_mean(qs.mean(axis=-1), mask, axis_name),
        )

    def actor_loss(self, agent_state, sample_batch, key, agent_id, axis_name=None):
        models = self.get_models(agent_state)
        features = local_features(self, agent_state, sample_batch.obs)
        index = self.agents.index(agent_id)
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
        q = models[agent_id].critic(inputs, joint_actions(self, actions))[..., 0]
        return PyTreeDict(
            actor_loss=-masked_mean(q, sample_batch.extras.active[agent_id], axis_name)
        )


class MATD3Workflow(MADDPGWorkflow):
    """Use joint replay with actor/target updates every `policy_delay` critic steps.

    Collection and initialization follow the continuous-action MADDPG layout.
    Each algorithm owns its update loop; delaying updates counts optimizer
    steps, independently of rollout length or updates per workflow iteration.
    """

    agent_class = MATD3Agent

    @classmethod
    def name(cls):
        return "MATD3"

    @classmethod
    def _build_from_config(cls, config):
        if config.policy_delay <= 0:
            raise ValueError("policy_delay must be positive")
        return super()._build_from_config(config)

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
            critic_loss_total = jnp.float32(0)
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
                critic_loss_total += value
            critic_state = agent_state.replace(model_state=nnx.unpack(models))
            critic_updates = nnx.merge(
                opt_state[self.agent.agents[0]].critic, copy=True
            ).step[...]

            def update_actors(carry):
                agent_state, opt_state = carry
                models = nnx.merge(agent_state.model_state, copy=True)
                actor_losses = PyTreeDict()
                for name in self.agent.agents:
                    auxiliary = nnx.clone(
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
                            path[:2] == (name, "actor")
                            and isinstance(variable, nnx.Param)
                        ),
                    )(models)
                    optimizer = nnx.merge(opt_state[name].actor, copy=True)
                    optimizer.update(models[name].actor, grads[name]["actor"])
                    opt_state = opt_state.replace(
                        **{name: opt_state[name].replace(actor=nnx.unpack(optimizer))}
                    )
                    nnx.update(models[name].critic, auxiliary)
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
                    actor_losses[name] = value
                return (
                    agent_state.replace(model_state=nnx.unpack(models)),
                    opt_state,
                    actor_losses,
                )

            def skip_actors(carry):
                return (
                    *carry,
                    PyTreeDict({name: jnp.float32(0) for name in self.agent.agents}),
                )

            actor_updated = critic_updates % self.config.policy_delay == 0
            agent_state, opt_state, actor_losses = jax.lax.cond(
                actor_updated, update_actors, skip_actors, (critic_state, opt_state)
            )
            losses = PyTreeDict(
                {
                    name: losses[name].replace(
                        actor_loss=actor_losses[name],
                        actor_updates=actor_updated.astype(jnp.float32),
                    )
                    for name in self.agent.agents
                }
            )
            return (agent_state, opt_state, key), (critic_loss_total, losses)

        (agent_state, opt_state, _), (critic_loss_totals, losses) = jax.lax.scan(
            update,
            (agent_state, state.opt_state, learn_key),
            (),
            length=self.config.num_updates_per_iter,
        )
        losses = PyTreeDict(
            {
                name: info.replace(
                    critic_loss=info.critic_loss.mean(),
                    q_value=info.q_value.mean(),
                    actor_loss=info.actor_loss.sum()
                    / jnp.maximum(info.actor_updates.sum(), 1),
                    actor_updates=info.actor_updates.sum(),
                )
                for name, info in losses.items()
            }
        )
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
            loss=critic_loss_totals.mean()
            + sum(info.actor_loss for info in losses.values()),
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
