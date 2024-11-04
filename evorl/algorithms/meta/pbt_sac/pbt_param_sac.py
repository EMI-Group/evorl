import logging
import math
from omegaconf import DictConfig

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.sharding import NamedSharding, PartitionSpec as P

from evorl.agent import RandomAgent
from evorl.distributed import POP_AXIS_NAME, tree_device_get, parallel_map
from evorl.utils.rl_toolkits import flatten_rollout_trajectory
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.types import PyTreeDict, State
from evorl.metrics import MetricBase
from evorl.rollout import rollout
from evorl.workflows import RLWorkflow
from evorl.utils.jax_utils import scan_and_last

from ..pbt_base import PBTTrainMetric
from ..pbt import PBTWorkflowTemplate
from ..utils import uniform_init, log_uniform_init
from ...offpolicy_utils import clean_trajectory

logger = logging.getLogger(__name__)


class PBTOffpolicyTrainMetric(PBTTrainMetric):
    rb_size: chex.Array


class PBTParamSACWorkflow(PBTWorkflowTemplate):
    def __init__(self, workflow: RLWorkflow, config: DictConfig):
        super().__init__(workflow, config)
        self.replay_buffer = workflow.replay_buffer

    @classmethod
    def name(cls):
        return "PBT-ParamSAC"

    @classmethod
    def build_from_config(
        cls, config: DictConfig, enable_multi_devices=True, enable_jit: bool = True
    ):
        num_devices = jax.local_device_count()
        if num_devices > 1:
            # Note: this ensures shard_map is not used
            raise ValueError("PBT-ParamSAC does not support multi-devices.")

        workflow = super().build_from_config(config, enable_multi_devices, enable_jit)
        workflow.replay_buffer = workflow.workflow.replay_buffer
        return workflow

    def _setup_pop(self, key: chex.PRNGKey) -> chex.ArrayTree:
        search_space = self.config.search_space

        def _init(hp, key):
            match hp:
                case "discount_g" | "log_alpha":
                    return uniform_init(search_space[hp], key, self.config.pop_size)
                case "actor_loss_weight" | "critic_loss_weight":
                    return log_uniform_init(search_space[hp], key, self.config.pop_size)

        pop = PyTreeDict(
            {
                hp: _init(hp, key)
                for hp, key in zip(
                    search_space.keys(), jax.random.split(key, len(search_space))
                )
            }
        )

        return pop

    def setup(self, key: chex.PRNGKey):
        key, rb_key = jax.random.split(key)
        state = super().setup(key)

        state = state.replace(
            replay_buffer_state=self.workflow._setup_replaybuffer(rb_key),
        )

        logger.info("Start replay buffer post-setup")
        state = self._postsetup_replaybuffer(state)

        logger.info("Complete replay buffer post-setup")

        return state

    def _postsetup_replaybuffer(self, state: State) -> State:
        env = self.workflow.env
        action_space = env.action_space
        obs_space = env.obs_space
        config = self.config.target_workflow
        replay_buffer_state = state.replay_buffer_state

        def _rollout(agent, agent_state, key, rollout_length):
            env_key, rollout_key = jax.random.split(key)

            env_state = env.reset(env_key)

            trajectory, env_state = rollout(
                env_fn=env.step,
                action_fn=agent.compute_actions,
                env_state=env_state,
                agent_state=agent_state,
                key=rollout_key,
                rollout_length=rollout_length,
                env_extra_fields=("ori_obs", "termination"),
            )

            # [T, B, ...] -> [T*B, ...]
            trajectory = clean_trajectory(trajectory)
            trajectory = flatten_rollout_trajectory(trajectory)
            trajectory = tree_stop_gradient(trajectory)

            return trajectory

        # ==== fill random transitions ====

        key, random_rollout_key, rollout_key = jax.random.split(state.key, num=3)
        random_agent = RandomAgent()
        random_agent_state = random_agent.init(
            obs_space, action_space, jax.random.PRNGKey(0)
        )
        rollout_length = config.random_timesteps // config.num_envs

        trajectory = _rollout(
            random_agent,
            random_agent_state,
            key=random_rollout_key,
            rollout_length=rollout_length,
        )

        # TODO: add support for shared obs_preprocessor and init it.

        replay_buffer_state = self.replay_buffer.add(replay_buffer_state, trajectory)

        sampled_timesteps_m = rollout_length * config.num_envs / 1e6

        workflow_metrics = state.metrics.replace(
            sampled_timesteps_m=state.metrics.sampled_timesteps_m + sampled_timesteps_m,
        )

        return state.replace(
            key=key,
            metrics=workflow_metrics,
            replay_buffer_state=replay_buffer_state,
        )

    def step(self, state: State) -> tuple[MetricBase, State]:
        pop_workflow_state = state.pop_workflow_state
        pop = state.pop
        replay_buffer_state = state.replay_buffer_state

        # ===== step ======
        def _train_steps(pop_wf_state, replay_buffer_state):
            def _one_step(carry, _):
                pop_wf_state, replay_buffer_state = carry

                def _wf_step_wrapper(wf_state):
                    wf_state = wf_state.replace(replay_buffer_state=replay_buffer_state)
                    train_metrics, wf_state = self.workflow.step(wf_state)
                    wf_state = wf_state.replace(replay_buffer_state=None)
                    return train_metrics, wf_state

                pop_train_metrics, pop_wf_state = jax.vmap(
                    _wf_step_wrapper, spmd_axis_name=POP_AXIS_NAME
                )(pop_wf_state)

                # add replay buffer data:
                # [pop, T*B, ...] -> [pop*T*B, ...]
                trajectory = jtu.tree_map(
                    lambda x: jax.lax.collapse(x, 0, 2), pop_train_metrics.trajectory
                )

                replay_buffer_state = self.replay_buffer.add(
                    replay_buffer_state, trajectory
                )
                pop_train_metrics = pop_train_metrics.replace(trajectory=None)

                return (pop_wf_state, replay_buffer_state), pop_train_metrics

            (pop_wf_state, replay_buffer_state), train_metrics = scan_and_last(
                _one_step,
                (pop_wf_state, replay_buffer_state),
                (),
                length=self.config.workflow_steps_per_iter,
            )

            return train_metrics, pop_wf_state, replay_buffer_state

        if self.config.parallel_train:
            sharding = self.sharding
            share_sharding = NamedSharding(sharding.mesh, P())

            pop_train_metrics, pop_workflow_state, replay_buffer_state = jax.jit(
                _train_steps,
                in_shardings=(sharding, share_sharding),
                out_shardings=(sharding, sharding, share_sharding),
            )(pop_workflow_state, replay_buffer_state)

        else:
            raise NotImplementedError(
                "PBT-ParamSAC does not support sequential train yet."
            )

        # ===== eval ======
        if self.config.parallel_eval:
            eval_fn = jax.vmap(self.workflow.evaluate, spmd_axis_name=POP_AXIS_NAME)
        else:
            eval_fn = parallel_map(self.workflow.evaluate, self.sharding)

        pop_eval_metrics, pop_workflow_state = jax.jit(
            eval_fn, in_shardings=self.sharding, out_shardings=self.sharding
        )(pop_workflow_state)

        # customize your pop metrics here
        pop_episode_returns = pop_eval_metrics.episode_returns

        # ===== warmup or exploit & explore ======
        key, exploit_and_explore_key = jax.random.split(state.key)

        def _dummy_fn(pop, pop_workflow_state, pop_metrics, key):
            return pop, pop_workflow_state

        pop, pop_workflow_state = jax.lax.cond(
            state.metrics.iterations + 1
            <= math.ceil(
                self.config.warmup_steps / self.config.workflow_steps_per_iter
            ),
            _dummy_fn,
            self.exploit_and_explore,
            pop,
            pop_workflow_state,
            pop_episode_returns,
            exploit_and_explore_key,
        )

        # ===== record metrics ======
        workflow_metrics = state.metrics.replace(
            sampled_timesteps_m=jnp.sum(
                pop_workflow_state.metrics.sampled_timesteps / 1e6
            ),  # convert uint32 to float32
            iterations=state.metrics.iterations + 1,
        )

        train_metrics = PBTOffpolicyTrainMetric(
            pop_episode_returns=pop_eval_metrics.episode_returns,
            pop_episode_lengths=pop_eval_metrics.episode_lengths,
            pop_train_metrics=pop_train_metrics,
            pop=state.pop,  # save prev pop instead of new pop to match the metrics
            rb_size=replay_buffer_state.buffer_size,
        )

        train_metrics = tree_device_get(train_metrics, self.devices[0])

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            pop=pop,
            pop_workflow_state=pop_workflow_state,
            replay_buffer_state=replay_buffer_state,
        )

    def apply_hyperparams_to_workflow_state(
        self, workflow_state: State, hyperparams: PyTreeDict[str, chex.Numeric]
    ):
        agent_state = workflow_state.agent_state
        agent_state = agent_state.replace(
            params=agent_state.params.replace(
                log_alpha=hyperparams.log_alpha,
            ),
            extra_state=agent_state.extra_state.replace(
                discount_g=hyperparams.discount_g,
            ),
        )

        # make a shadow copy
        hyperparams = hyperparams.replace()
        hyperparams.pop("discount_g")
        hyperparams.pop("log_alpha")

        return workflow_state.replace(
            agent_state=agent_state,
            hp_state=hyperparams,
        )

    @classmethod
    def enable_jit(cls) -> None:
        super().enable_jit()
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )
