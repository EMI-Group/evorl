import copy
import logging
import math
from functools import partial
from omegaconf import DictConfig, OmegaConf, open_dict, read_write

import chex
import hydra
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
import orbax.checkpoint as ocp


from evorl.agent import RandomAgent
from evorl.distributed import (
    POP_AXIS_NAME,
    parallel_map,
    tree_device_get,
    tree_device_put,
)
from evorl.rollout import rollout
from evorl.metrics import MetricBase
from evorl.types import MISSING_REWARD, PyTreeDict, State
from evorl.recorders import get_1d_array_statistics
from evorl.utils.rl_toolkits import flatten_rollout_trajectory
from evorl.utils.jax_utils import (
    tree_get,
    tree_set,
    tree_stop_gradient,
    scan_and_last,
    is_jitted,
)
from evorl.workflows import RLWorkflow, OffPolicyWorkflow, Workflow

from .pbt_utils import convert_pop_to_df
from .pbt_operations import explore, select
from ..offpolicy_utils import clean_trajectory, skip_replay_buffer_state

logger = logging.getLogger(__name__)


class PBTTrainMetric(MetricBase):
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array
    pop_train_metrics: MetricBase
    pop: chex.ArrayTree


class PBTOffpolicyTrainMetric(PBTTrainMetric):
    rb_size: chex.Array


class PBTEvalMetric(MetricBase):
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array


class PBTWorkflowMetric(MetricBase):
    # the average of sampled timesteps of all workflows
    sampled_timesteps_m: chex.Array = jnp.zeros((), dtype=jnp.float32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class PBTWorkflowBase(Workflow):
    def __init__(self, workflow: RLWorkflow, config: DictConfig):
        super().__init__(config)

        self.workflow = workflow
        self.devices = jax.local_devices()[:1]
        self.sharding = None  # training sharding
        # self.pbt_update_sharding = None

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        num_devices = jax.device_count()

        if config.pop_size % num_devices != 0:
            logger.warning(
                f"pop_size({config.pop_size}) cannot be divided by num_devices({num_devices}), "
                f"rescale pop_size to {config.pop_size // num_devices * num_devices}"
            )

        config.pop_size = (config.pop_size // num_devices) * num_devices

    @classmethod
    def build_from_config(
        cls, config: DictConfig, enable_multi_devices=True, enable_jit: bool = True
    ):
        config = copy.deepcopy(config)  # avoid in-place modification

        devices = jax.local_devices()

        OmegaConf.set_readonly(config, False)
        cls._rescale_config(config)

        if enable_jit:
            cls.enable_jit()

        OmegaConf.set_readonly(config, True)

        workflow = cls._build_from_config(config)

        mesh = Mesh(devices, axis_names=(POP_AXIS_NAME,))
        workflow.devices = devices
        workflow.sharding = NamedSharding(mesh, P(POP_AXIS_NAME))

        return workflow

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        target_workflow_config = config.target_workflow
        target_workflow_config = copy.deepcopy(target_workflow_config)
        target_workflow_cls = hydra.utils.get_class(target_workflow_config.workflow_cls)

        devices = jax.local_devices()

        with read_write(target_workflow_config):
            with open_dict(target_workflow_config):
                target_workflow_config.env = copy.deepcopy(config.env)
                # disable target workflow ckpt
                target_workflow_config.checkpoint = OmegaConf.create(dict(enable=False))

        OmegaConf.set_readonly(target_workflow_config, True)

        enable_jit = is_jitted(cls.step)
        target_workflow = target_workflow_cls.build_from_config(
            target_workflow_config, enable_jit=enable_jit
        )

        target_workflow.devices = devices

        return cls(target_workflow, config)

    def _setup_pop(self, key: chex.PRNGKey) -> chex.ArrayTree:
        raise NotImplementedError

    def _customize_optimizer(self) -> None:
        pass

    def setup(self, key: chex.PRNGKey):
        pop_size = self.config.pop_size
        self._customize_optimizer()

        key, workflow_key, pop_key = jax.random.split(key, num=3)

        pop = self._setup_pop(pop_key)
        pop = tree_device_put(pop, self.sharding)

        # save metric on GPU0
        workflow_metrics = PBTWorkflowMetric()

        workflow_keys = jax.random.split(workflow_key, pop_size)
        workflow_keys = jax.device_put(workflow_keys, self.sharding)

        pop_workflow_state = jax.jit(
            jax.vmap(self.workflow.setup, spmd_axis_name=POP_AXIS_NAME),
            in_shardings=self.sharding,
            out_shardings=self.sharding,
        )(workflow_keys)

        pop_workflow_state = jax.jit(
            jax.vmap(
                self.apply_hyperparams_to_workflow_state, spmd_axis_name=POP_AXIS_NAME
            ),
            in_shardings=self.sharding,
            out_shardings=self.sharding,
        )(pop_workflow_state, pop)

        return State(
            key=key,
            metrics=workflow_metrics,
            pop_workflow_state=pop_workflow_state,
            pop=pop,
        )

    def step(self, state: State) -> tuple[MetricBase, State]:
        pop_workflow_state = state.pop_workflow_state
        pop = state.pop

        # ===== step ======
        def _train_steps(wf_state):
            def _one_step(wf_state, _):
                train_metrics, wf_state = self.workflow.step(wf_state)
                return wf_state, train_metrics

            wf_state, train_metrics = scan_and_last(
                _one_step, wf_state, (), length=self.config.workflow_steps_per_iter
            )

            return train_metrics, wf_state

        if self.config.parallel_train:
            train_steps_fn = jax.vmap(_train_steps, spmd_axis_name=POP_AXIS_NAME)
        else:
            # TODO: fix potential unneccesary gpu-comm: eg: all-gather in ppo
            # train_steps_fn = partial(jax.lax.map, _train_steps)
            train_steps_fn = parallel_map(_train_steps, self.sharding)

        pop_train_metrics, pop_workflow_state = jax.jit(
            train_steps_fn, in_shardings=self.sharding, out_shardings=self.sharding
        )(pop_workflow_state)

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

        train_metrics = PBTTrainMetric(
            pop_episode_returns=pop_eval_metrics.episode_returns,
            pop_episode_lengths=pop_eval_metrics.episode_lengths,
            pop_train_metrics=pop_train_metrics,
            pop=state.pop,  # save prev pop instead of new pop to match the metrics
        )

        train_metrics = tree_device_get(train_metrics, self.devices[0])

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            pop=pop,
            pop_workflow_state=pop_workflow_state,
        )

    def evaluate(self, state: State) -> State:
        # Tips: evaluation consumes every workflow_state's internal key
        if self.config.parallel_eval:
            eval_fn = jax.vmap(self.workflow.evaluate, spmd_axis_name=POP_AXIS_NAME)
        else:
            eval_fn = parallel_map(self.workflow.evaluate, self.sharding)

        pop_eval_metrics, pop_workflow_state = jax.jit(
            eval_fn, in_shardings=self.sharding, out_shardings=self.sharding
        )(state.pop_workflow_state)

        # customize your pop metrics here
        eval_metrics = PBTEvalMetric(
            pop_episode_returns=pop_eval_metrics.episode_returns,
            pop_episode_lengths=pop_eval_metrics.episode_lengths,
        )

        return eval_metrics, state.replace(pop_workflow_state=pop_workflow_state)

    def exploit_and_explore(
        self,
        pop: chex.ArrayTree,
        pop_workflow_state: State,
        pop_metrics: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.ArrayTree, State]:
        raise NotImplementedError

    def apply_hyperparams_to_workflow_state(
        self, workflow_state: State, hyperparams: PyTreeDict[str, chex.Numeric]
    ) -> State:
        raise NotImplementedError

    @classmethod
    def enable_jit(cls) -> None:
        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls.step = jax.jit(cls.step, static_argnums=(0,))


class PBTWorkflowTemplate(PBTWorkflowBase):
    """
    Standard PBT Workflow Template
    """

    def exploit_and_explore(
        self,
        pop: chex.ArrayTree,
        pop_workflow_state: State,
        pop_metrics: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.ArrayTree, State]:
        exploit_key, explore_key = jax.random.split(key)

        config = self.config

        tops_indices, bottoms_indices = select(
            pop_metrics,  # using episode_return
            exploit_key,
            bottoms_num=round(config.pop_size * config.bottom_ratio),
            tops_num=round(config.pop_size * config.top_ratio),
        )

        parents = tree_get(pop, tops_indices)
        parents_wf_state = tree_get(pop_workflow_state, tops_indices)

        # TODO: check sharding issue with vmap under multi-devices.
        offsprings = jax.vmap(
            partial(
                explore,
                perturb_factor=config.perturb_factor,
                search_space=config.search_space,
            )
        )(parents, jax.random.split(explore_key, bottoms_indices.shape[0]))

        # Note: no need to deepcopy parents_wf_state here, since it should be
        # ensured immutable in apply_hyperparams_to_workflow_state()
        offsprings_workflow_state = jax.vmap(self.apply_hyperparams_to_workflow_state)(
            parents_wf_state, offsprings
        )

        # ==== survival | merge population ====
        pop = tree_set(pop, offsprings, bottoms_indices, unique_indices=True)
        # we copy wf_state back to offspring wf_state
        pop_workflow_state = tree_set(
            pop_workflow_state,
            offsprings_workflow_state,
            bottoms_indices,
            unique_indices=True,
        )

        return pop, pop_workflow_state

    def learn(self, state: State) -> State:
        for i in range(state.metrics.iterations, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            self.recorder.write(workflow_metrics.to_local_dict(), iters)

            train_metrics_dict = train_metrics.to_local_dict()

            if "train_episode_return" in train_metrics_dict["pop_train_metrics"]:
                train_episode_return = train_metrics_dict["pop_train_metrics"][
                    "train_episode_return"
                ]
                train_metrics_dict["pop_train_metrics"]["train_episode_return"] = (
                    train_episode_return[train_episode_return != MISSING_REWARD]
                )

            train_metrics_dict["pop_episode_returns"] = get_1d_array_statistics(
                train_metrics_dict["pop_episode_returns"], histogram=True
            )
            train_metrics_dict["pop_episode_lengths"] = get_1d_array_statistics(
                train_metrics_dict["pop_episode_lengths"], histogram=True
            )

            train_metrics_dict["pop"] = convert_pop_to_df(train_metrics_dict["pop"])

            train_metrics_dict["pop_train_metrics"] = jtu.tree_map(
                get_1d_array_statistics, train_metrics_dict["pop_train_metrics"]
            )

            self.recorder.write(train_metrics_dict, iters)

            self.checkpoint_manager.save(iters, args=ocp.args.StandardSave(state))

        return state


class PBTOffpolicyWorkflowTemplate(PBTWorkflowTemplate):
    """
    PBT Workflow Template for Off-policy algorithms with shared replay buffer
    """

    def __init__(self, workflow: OffPolicyWorkflow, config: DictConfig):
        super().__init__(workflow, config)
        self.replay_buffer = workflow.replay_buffer

    @classmethod
    def build_from_config(
        cls, config: DictConfig, enable_multi_devices=True, enable_jit: bool = True
    ):
        num_devices = jax.local_device_count()
        if num_devices > 1:
            # Note: this ensures shard_map is not used
            raise ValueError(
                "PBTOffpolicyWorkflowTemplate does not support multi-devices yet."
            )

        return super().build_from_config(config, enable_multi_devices, enable_jit)

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
            # TODO: impl it
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

    def learn(self, state: State) -> State:
        for i in range(state.metrics.iterations, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            self.recorder.write(workflow_metrics.to_local_dict(), iters)

            train_metrics_dict = train_metrics.to_local_dict()

            train_metrics_dict["pop_episode_returns"] = get_1d_array_statistics(
                train_metrics_dict["pop_episode_returns"], histogram=True
            )
            train_metrics_dict["pop_episode_lengths"] = get_1d_array_statistics(
                train_metrics_dict["pop_episode_lengths"], histogram=True
            )

            train_metrics_dict["pop"] = convert_pop_to_df(train_metrics_dict["pop"])

            train_metrics_dict["pop_train_metrics"] = jtu.tree_map(
                get_1d_array_statistics, train_metrics_dict["pop_train_metrics"]
            )

            self.recorder.write(train_metrics_dict, iters)

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)
            self.checkpoint_manager.save(
                iters,
                args=ocp.args.StandardSave(saved_state),
            )

        return state

    @classmethod
    def enable_jit(cls) -> None:
        super().enable_jit()
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )
