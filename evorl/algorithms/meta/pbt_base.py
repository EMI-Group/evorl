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
from jax.experimental.shard_map import shard_map
import orbax.checkpoint as ocp


from evorl.agent import RandomAgent
from evorl.distributed import (
    POP_AXIS_NAME,
    shmap_vmap,
    shmap_map,
    tree_device_put,
)
from evorl.rollout import rollout
from evorl.metrics import MetricBase, EvaluateMetric
from evorl.envs import AutoresetMode, create_env, Discrete
from evorl.evaluators import Evaluator
from evorl.sample_batch import SampleBatch
from evorl.types import MISSING_REWARD, PyTreeDict, State
from evorl.replay_buffers import ReplayBufferState
from evorl.recorders import get_1d_array_statistics, get_1d_array, add_prefix
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
    def __init__(self, workflow: RLWorkflow, evaluator: Evaluator, config: DictConfig):
        super().__init__(config)

        self.workflow = workflow
        self.evaluator = evaluator
        self.devices = jax.local_devices()[:1]
        self.sharding = None  # training sharding

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

        eval_env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )
        evaluator = Evaluator(
            env=eval_env,
            action_fn=target_workflow.agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        return cls(target_workflow, evaluator, config)

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

        # save metrics on GPU0
        workflow_metrics = PBTWorkflowMetric()

        workflow_keys = jax.random.split(workflow_key, pop_size)
        workflow_keys = jax.device_put(workflow_keys, self.sharding)

        pop_workflow_state = shmap_vmap(
            self.workflow.setup,
            mesh=self.sharding.mesh,
            in_specs=self.sharding.spec,
            out_specs=self.sharding.spec,
            check_rep=False,
        )(workflow_keys)

        pop_workflow_state = shmap_vmap(
            self.apply_hyperparams_to_workflow_state,
            mesh=self.sharding.mesh,
            in_specs=self.sharding.spec,
            out_specs=self.sharding.spec,
            check_rep=False,
        )(pop_workflow_state, pop)

        return State(
            key=key,  # GPU0
            metrics=workflow_metrics,  # GPU0
            pop_workflow_state=pop_workflow_state,  # sharding across devices
            pop=pop,  # sharding across devices
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
            train_steps_fn = shmap_vmap(
                _train_steps,
                mesh=self.sharding.mesh,
                in_specs=self.sharding.spec,
                out_specs=self.sharding.spec,
                check_rep=False,
            )
        else:
            train_steps_fn = shmap_map(
                _train_steps,
                mesh=self.sharding.mesh,
                in_specs=self.sharding.spec,
                out_specs=self.sharding.spec,
                check_rep=False,
            )

        pop_train_metrics, pop_workflow_state = train_steps_fn(pop_workflow_state)

        # ===== eval ======
        if self.config.parallel_eval:
            eval_fn = shmap_vmap(
                self.workflow.evaluate,
                mesh=self.sharding.mesh,
                in_specs=self.sharding.spec,
                out_specs=self.sharding.spec,
                check_rep=False,
            )
        else:
            eval_fn = shmap_map(
                self.workflow.evaluate,
                mesh=self.sharding.mesh,
                in_specs=self.sharding.spec,
                out_specs=self.sharding.spec,
                check_rep=False,
            )

        pop_eval_metrics, pop_workflow_state = eval_fn(pop_workflow_state)

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

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            pop=pop,
            pop_workflow_state=pop_workflow_state,
        )

    def evaluate(self, state: State) -> State:
        key, eval_key = jax.random.split(state.key, num=2)

        def _evaluate(wf_state, key):
            # [#episodes]
            raw_eval_metrics = self.evaluator.evaluate(
                wf_state.agent_state, key, num_episodes=self.config.eval_episodes
            )

            eval_metrics = EvaluateMetric(
                episode_returns=raw_eval_metrics.episode_returns.mean(),
                episode_lengths=raw_eval_metrics.episode_lengths.mean(),
            )
            return eval_metrics

        if self.config.parallel_eval:
            eval_fn = shmap_vmap(
                _evaluate,
                mesh=self.sharding.mesh,
                in_specs=self.sharding.spec,
                out_specs=self.sharding.spec,
                check_rep=False,
            )
        else:
            eval_fn = shmap_map(
                _evaluate,
                mesh=self.sharding.mesh,
                in_specs=self.sharding.spec,
                out_specs=self.sharding.spec,
                check_rep=False,
            )

        pop_eval_metrics = eval_fn(
            state.pop_workflow_state, jax.random.split(eval_key, self.config.pop_size)
        )

        eval_metrics = PBTEvalMetric(
            pop_episode_returns=pop_eval_metrics.episode_returns,
            pop_episode_lengths=pop_eval_metrics.episode_lengths,
        )

        return eval_metrics, state.replace(key=key)

    def exploit_and_explore(
        self,
        pop: chex.ArrayTree,
        pop_workflow_state: State,
        pop_metrics: chex.ArrayTree,
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
        pop_metrics: chex.ArrayTree,
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

            self.recorder.write(workflow_metrics.to_local_dict(), iters)
            self.recorder.write(train_metrics_dict, iters)

            if iters % self.config.eval_interval == 0 or iters == self.config.num_iters:
                eval_metrics, state = self.evaluate(state)

                eval_metrics_dict = jtu.tree_map(
                    get_1d_array,
                    eval_metrics.to_local_dict(),
                )

                self.recorder.write(add_prefix(eval_metrics_dict, "eval"), iters)

            self.checkpoint_manager.save(iters, args=ocp.args.StandardSave(state))

        return state


class PBTOffpolicyWorkflowTemplate(PBTWorkflowTemplate):
    """
    PBT Workflow Template for Off-policy algorithms with shared replay buffer
    """

    def __init__(
        self, workflow: OffPolicyWorkflow, evaluator: Evaluator, config: DictConfig
    ):
        super().__init__(workflow, evaluator, config)
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
            replay_buffer_state=self._setup_replaybuffer(rb_key),
        )

        logger.info("Start replay buffer post-setup")
        state = self._postsetup_replaybuffer(state)

        logger.info("Complete replay buffer post-setup")

        return state

    def _setup_replaybuffer(self, key: chex.PRNGKey) -> ReplayBufferState:
        action_space = self.env.action_space
        obs_space = self.env.obs_space

        # create dummy data to initialize the replay buffer
        if isinstance(action_space, Discrete):
            dummy_action = jnp.zeros((), dtype=jnp.int32)
        else:
            dummy_action = jnp.zeros(action_space.shape)
        dummy_obs = jnp.zeros(obs_space.shape)

        dummy_reward = jnp.zeros(())
        dummy_done = jnp.zeros(())

        dummy_sample_batch = SampleBatch(
            obs=dummy_obs,
            actions=dummy_action,
            rewards=dummy_reward,
            # next_obs=dummy_obs,
            # dones=dummy_done,
            extras=PyTreeDict(
                policy_extras=PyTreeDict(),
                env_extras=PyTreeDict(
                    {"ori_obs": dummy_obs, "termination": dummy_done}
                ),
            ),
        )
        replay_buffer_state = self.replay_buffer.init(dummy_sample_batch)

        return replay_buffer_state

    def _postsetup_replaybuffer(self, state: State) -> State:
        env = self.workflow.env
        action_space = env.action_space
        obs_space = env.obs_space
        num_envs = self.config.target_workflow.num_envs
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
        rollout_length = self.config.random_timesteps // num_envs

        trajectory = _rollout(
            random_agent,
            random_agent_state,
            key=random_rollout_key,
            rollout_length=rollout_length,
        )

        # TODO: add support for shared obs_preprocessor and init it.

        replay_buffer_state = self.replay_buffer.add(replay_buffer_state, trajectory)

        sampled_timesteps_m = rollout_length * num_envs / 1e6

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

                pop_train_metrics, pop_wf_state = jax.vmap(_wf_step_wrapper)(
                    pop_wf_state
                )

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

        pop_train_metrics, pop_workflow_state, replay_buffer_state = shard_map(
            _train_steps,
            mesh=self.sharding.mesh,
            in_specs=(P(POP_AXIS_NAME), P()),
            out_specs=(P(POP_AXIS_NAME), P(POP_AXIS_NAME), P()),
        )(pop_workflow_state, replay_buffer_state)

        # ===== eval ======
        if self.config.parallel_eval:
            eval_fn = shmap_vmap(
                self.workflow.evaluate,
                mesh=self.sharding.mesh,
                in_specs=self.sharding.spec,
                out_specs=self.sharding.spec,
                check_rep=False,
            )
        else:
            eval_fn = shmap_map(
                self.workflow.evaluate,
                mesh=self.sharding.mesh,
                in_specs=self.sharding.spec,
                out_specs=self.sharding.spec,
                check_rep=False,
            )

        pop_eval_metrics, pop_workflow_state = eval_fn(pop_workflow_state)

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

            self.recorder.write(workflow_metrics.to_local_dict(), iters)
            self.recorder.write(train_metrics_dict, iters)

            if iters % self.config.eval_interval == 0 or iters == self.config.num_iters:
                eval_metrics, state = self.evaluate(state)

                eval_metrics_dict = jtu.tree_map(
                    get_1d_array,
                    eval_metrics.to_local_dict(),
                )

                self.recorder.write(add_prefix(eval_metrics_dict, "eval"), iters)

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
