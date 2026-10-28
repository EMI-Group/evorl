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


from evorl.agent import RandomAgent
from evorl.distributed import (
    META_POP_AXIS_NAME,
    all_gather,
    shmap_vmap,
    tree_scatter_set,
)
from evorl.rollout import rollout
from evorl.metrics import MetricBase, EvaluateMetric, metric_field
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators import Evaluator
from evorl.types import MISSING_REWARD, PyTreeDict, State, PyTreeData
from evorl.replay_buffers import ReplayBufferState
from evorl.recorders import get_1d_array_statistics, get_1d_array, add_prefix
from evorl.utils.rl_toolkits import flatten_rollout_trajectory
from evorl.utils.jax_utils import (
    tree_get,
    tree_stop_gradient,
    scan_and_last,
    is_jitted,
)
from evorl.utils import running_statistics
from evorl.workflows import RLWorkflow, OffPolicyWorkflow, Workflow

from .pbt_utils import convert_pop_to_df
from .pbt_operations import explore, select
from ..offpolicy_utils import clean_trajectory, skip_replay_buffer_state

logger = logging.getLogger(__name__)


class PBTTrainMetric(MetricBase):
    pop_episode_return: jax.Array
    pop_episode_length: jax.Array
    pop_train_metrics: MetricBase
    pop: chex.ArrayTree


class PBTOffpolicyTrainMetric(PBTTrainMetric):
    rb_size: jax.Array


class PBTEvalMetric(MetricBase):
    pop_episode_return: jax.Array
    pop_episode_length: jax.Array


class PBTWorkflowMetric(MetricBase):
    # Total environment samples across the population, in millions.
    sampled_timesteps_m: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.float32)
    )
    iterations: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )


class PBTOptState(PyTreeData):
    pass


class PBTWorkflowBase(Workflow):
    def __init__(self, workflow: RLWorkflow, evaluator: Evaluator, config: DictConfig):
        super().__init__(config)

        self.workflow = workflow
        self.evaluator = evaluator
        self.mesh = jax.make_mesh(
            (1,),
            (META_POP_AXIS_NAME,),
            axis_types=(jax.sharding.AxisType.Explicit,),
            devices=jax.local_devices()[:1],
        )

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        num_devices = jax.local_device_count()

        if config.pop_size < num_devices or config.pop_size % num_devices:
            raise ValueError(
                f"pop_size({config.pop_size}) must be divisible by num_devices({num_devices})"
            )

    @classmethod
    def build_from_config(cls, config: DictConfig, multi_device=True, jit: bool = True):
        config = copy.deepcopy(config)  # avoid in-place modification

        devices = jax.local_devices() if multi_device else jax.local_devices()[:1]

        OmegaConf.set_readonly(config, False)
        if multi_device:
            cls._rescale_config(config)

        if jit:
            cls.enable_jit()

        OmegaConf.set_readonly(config, True)

        workflow = cls._build_from_config(config)

        mesh = jax.make_mesh(
            (len(devices),),
            (META_POP_AXIS_NAME,),
            axis_types=(jax.sharding.AxisType.Explicit,),
            devices=devices,
        )
        workflow.mesh = mesh

        return workflow

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        target_workflow_config = config.target_workflow
        target_workflow_config = copy.deepcopy(target_workflow_config)
        target_workflow_cls = hydra.utils.get_class(target_workflow_config.workflow_cls)

        with read_write(target_workflow_config), open_dict(target_workflow_config):
            target_workflow_config.env = copy.deepcopy(config.env)
            # disable target workflow ckpt
            target_workflow_config.checkpoint = OmegaConf.create(dict(enable=False))

        OmegaConf.set_readonly(target_workflow_config, True)

        jit = is_jitted(cls.step)
        target_workflow = target_workflow_cls.build_from_config(
            target_workflow_config, jit=jit
        )

        eval_env = create_env(
            config.env,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )
        evaluator = Evaluator(
            env=eval_env,
            action_fn=target_workflow.agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        return cls(target_workflow, evaluator, config)

    def _setup_pop_and_pbt_optimizer(
        self, key: jax.Array
    ) -> tuple[chex.ArrayTree, PBTOptState]:
        raise NotImplementedError

    def _customize_optimizer(self) -> None:
        pass

    def setup(self, key: jax.Array):
        pop_size = self.config.pop_size
        self._customize_optimizer()

        key, workflow_key, pop_key = jax.random.split(key, num=3)

        pop, pbt_opt_state = self._setup_pop_and_pbt_optimizer(pop_key)
        pop = jax.device_put(pop, jax.P(META_POP_AXIS_NAME))

        workflow_metrics = PBTWorkflowMetric()
        key, workflow_metrics, pbt_opt_state = jax.device_put(
            (key, workflow_metrics, pbt_opt_state), jax.P()
        )

        workflow_keys = jax.random.split(workflow_key, pop_size)
        workflow_keys = jax.device_put(workflow_keys, jax.P(META_POP_AXIS_NAME))
        pop_workflow_state = shmap_vmap(
            self.workflow.setup,
            in_specs=jax.P(META_POP_AXIS_NAME),
            out_specs=jax.P(META_POP_AXIS_NAME),
        )(workflow_keys)

        # Note: for obs_preprocessor, we assume pop_workflow_state.agent_state.obs_preprocessor_state
        # is already same by initialization in self.workflow.setup(),
        # so we don't need sync them here.
        # Caution: for off-policy workflow with postsetup, this may not be true.

        pop_workflow_state = shmap_vmap(
            self.apply_hyperparams_to_workflow_state,
            in_specs=jax.P(META_POP_AXIS_NAME),
            out_specs=jax.P(META_POP_AXIS_NAME),
        )(pop_workflow_state, pop)

        return State(
            key=key,  # shared
            metrics=workflow_metrics,  # shared
            pop_workflow_state=pop_workflow_state,  # across devices
            pop=pop,  # across devices
            pbt_opt_state=pbt_opt_state,  # shared
        )

    def step(self, state: State) -> tuple[MetricBase, State]:
        pop_workflow_state = state.pop_workflow_state
        pop = state.pop
        pbt_opt_state = state.pbt_opt_state

        # ===== step ======
        def _train_steps(pop_wf_state):
            def _one_step(pop_wf_state, _):
                train_metrics, pop_wf_state = jax.vmap(self.workflow.step)(pop_wf_state)
                return pop_wf_state, train_metrics

            pop_wf_state, train_metrics = scan_and_last(
                _one_step, pop_wf_state, (), length=self.config.workflow_steps_per_iter
            )

            return train_metrics, pop_wf_state

        train_steps_fn = jax.shard_map(
            _train_steps,
            in_specs=jax.P(META_POP_AXIS_NAME),
            out_specs=jax.P(META_POP_AXIS_NAME),
        )

        pop_train_metrics, pop_workflow_state = train_steps_fn(pop_workflow_state)

        # ===== eval ======
        eval_fn = shmap_vmap(
            self.workflow.evaluate,
            in_specs=jax.P(META_POP_AXIS_NAME),
            out_specs=jax.P(META_POP_AXIS_NAME),
        )

        pop_eval_metrics, pop_workflow_state = eval_fn(pop_workflow_state)

        # customize your pop metrics here
        pop_episode_returns = pop_eval_metrics.episode_return

        # ===== warmup or exploit & explore ======
        key, exploit_and_explore_key = jax.random.split(state.key)

        def _dummy_fn(pbt_opt_state, pop, pop_workflow_state, pop_metrics, key):
            return pop, pop_workflow_state, pbt_opt_state

        pop, pop_workflow_state, pbt_opt_state = jax.lax.cond(
            state.metrics.iterations + 1
            <= math.ceil(
                self.config.warmup_steps / self.config.workflow_steps_per_iter
            ),
            _dummy_fn,
            self.exploit_and_explore,
            pbt_opt_state,
            pop,
            pop_workflow_state,
            pop_episode_returns,
            exploit_and_explore_key,
        )

        # ===== record metrics ======
        if hasattr(pop_workflow_state.metrics, "sampled_timesteps"):
            sampled_timesteps_m = jnp.sum(
                pop_workflow_state.metrics.sampled_timesteps / 1e6
            )
        elif hasattr(pop_workflow_state.metrics, "sampled_timesteps_m"):
            sampled_timesteps_m = jnp.sum(
                pop_workflow_state.metrics.sampled_timesteps_m
            )
        else:
            sampled_timesteps_m = jnp.zeros((), dtype=jnp.float32)

        # Note: sampled_timesteps_m is already accumulated in target_workflow
        workflow_metrics = state.metrics.replace(
            sampled_timesteps_m=sampled_timesteps_m,
            iterations=state.metrics.iterations + 1,
        )

        train_metrics = PBTTrainMetric(
            pop_episode_return=pop_eval_metrics.episode_return,
            pop_episode_length=pop_eval_metrics.episode_length,
            pop_train_metrics=pop_train_metrics,
            pop=state.pop,  # save prev pop instead of new pop to match the metrics
        )

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            pop=pop,
            pop_workflow_state=pop_workflow_state,
            pbt_opt_state=pbt_opt_state,
        )

    def evaluate(self, state: State) -> State:
        key, eval_key = jax.random.split(state.key, num=2)

        def _evaluate(wf_state, key):
            # [#episodes]
            raw_eval_metrics = self.evaluator.evaluate(
                wf_state.agent_state, key, num_episodes=self.config.eval_episodes
            )

            eval_metrics = EvaluateMetric(
                episode_return=raw_eval_metrics.episode_return.mean(axis=0),
                episode_length=raw_eval_metrics.episode_length.mean(),
            )
            return eval_metrics

        eval_fn = shmap_vmap(
            _evaluate,
            in_specs=jax.P(META_POP_AXIS_NAME),
            out_specs=jax.P(META_POP_AXIS_NAME),
        )

        pop_eval_metrics = eval_fn(
            state.pop_workflow_state,
            jax.device_put(
                jax.random.split(eval_key, self.config.pop_size),
                jax.P(META_POP_AXIS_NAME),
            ),
        )

        eval_metrics = PBTEvalMetric(
            pop_episode_return=pop_eval_metrics.episode_return,
            pop_episode_length=pop_eval_metrics.episode_length,
        )

        return eval_metrics, state.replace(key=key)

    def exploit_and_explore(
        self,
        pbt_opt_state: PBTOptState,
        pop: chex.ArrayTree,
        pop_workflow_state: State,
        pop_metrics: chex.ArrayTree,
        key: jax.Array,
    ) -> tuple[chex.ArrayTree, State, PBTOptState]:
        raise NotImplementedError

    def apply_hyperparams_to_workflow_state(
        self, workflow_state: State, hyperparams: PyTreeDict[str, jax.typing.ArrayLike]
    ) -> State:
        raise NotImplementedError

    @classmethod
    def enable_jit(cls) -> None:
        cls.setup = jax.jit(cls.setup, static_argnums=(0,))
        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls.step = jax.jit(cls.step, static_argnums=(0,))


class PBTWorkflowTemplate(PBTWorkflowBase):
    """Standard PBT Workflow Template."""

    def exploit_and_explore(
        self,
        pbt_opt_state: PBTOptState,
        pop: chex.ArrayTree,
        pop_workflow_state: State,
        pop_metrics: chex.ArrayTree,
        key: jax.Array,
    ) -> tuple[chex.ArrayTree, State, PBTOptState]:
        exploit_key, explore_key = jax.random.split(key)

        config = self.config
        # Sort global fitnesses; only this small vector is replicated.
        pop_metrics = jax.device_put(pop_metrics, jax.P())

        top_indices, bottom_indices = select(
            pop_metrics,  # using episode_return
            exploit_key,
            bottoms_num=round(config.pop_size * config.bottom_ratio),
            tops_num=round(config.pop_size * config.top_ratio),
        )

        parents = tree_get(pop, top_indices, out_sharding=jax.P())
        parents_wf_state = tree_get(
            pop_workflow_state, top_indices, out_sharding=jax.P()
        )

        offsprings = jax.vmap(
            partial(
                explore,
                perturb_factor=config.perturb_factor,
                search_space=config.search_space,
            )
        )(parents, jax.random.split(explore_key, bottom_indices.shape[0]))

        # Note: no need to deepcopy parents_wf_state here, since it should be
        # ensured immutable in apply_hyperparams_to_workflow_state()
        offsprings_workflow_state = jax.vmap(self.apply_hyperparams_to_workflow_state)(
            parents_wf_state, offsprings
        )
        # Preserve independent exploration streams after copying a parent's
        # model and environment snapshot into the destination member.
        offsprings_workflow_state = offsprings_workflow_state.replace(
            key=tree_get(pop_workflow_state.key, bottom_indices, out_sharding=jax.P())
        )

        # ==== survival | merge population ====
        pop = tree_scatter_set(
            pop, offsprings, bottom_indices, jax.P(META_POP_AXIS_NAME)
        )
        # we copy wf_state back to offspring wf_state
        pop_workflow_state = tree_scatter_set(
            pop_workflow_state,
            offsprings_workflow_state,
            bottom_indices,
            jax.P(META_POP_AXIS_NAME),
        )

        return pop, pop_workflow_state, pbt_opt_state

    def _record_step_metrics(self, train_metrics, workflow_metrics, iters):
        train_metrics_dict = train_metrics.to_local_dict()

        pop_train_metric = train_metrics_dict["pop_train_metrics"]
        if "train_episode_return" in pop_train_metric:
            train_episode_return = pop_train_metric["train_episode_return"]
            # Note: the order does not matter, since we use
            train_episode_return = train_episode_return[
                train_episode_return != MISSING_REWARD
            ]

            if len(train_episode_return) == 0:
                train_episode_return = None

            pop_train_metric["train_episode_return"] = train_episode_return

        train_metrics_dict.update(
            pop_episode_return=get_1d_array_statistics(
                train_metrics_dict["pop_episode_return"], histogram=True
            ),
            pop_episode_length=get_1d_array_statistics(
                train_metrics_dict["pop_episode_length"], histogram=True
            ),
            pop=convert_pop_to_df(train_metrics_dict["pop"]),
            pop_train_metrics=jtu.tree_map(get_1d_array_statistics, pop_train_metric),
        )

        self.recorder.write(workflow_metrics.to_local_dict(), iters)
        self.recorder.write(train_metrics_dict, iters)

    def learn(self, state: State) -> State:
        for i in range(state.metrics.iterations, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            self._record_step_metrics(train_metrics, workflow_metrics, iters)

            if iters % self.config.eval_interval == 0 or iters == self.config.num_iters:
                eval_metrics, state = self.evaluate(state)

                eval_metrics_dict = jtu.tree_map(
                    get_1d_array,
                    eval_metrics.to_local_dict(),
                )

                self.recorder.write(add_prefix(eval_metrics_dict, "eval"), iters)

            self.checkpoint_manager.save(
                iters,
                state,
                force=iters == self.config.num_iters,
            )

        return state


class PBTOffpolicyWorkflowTemplate(PBTWorkflowTemplate):
    """Train PBT members with a shared replay or independent per-member rings.

    With `shared_replay_buffer=True`, one replay ring is replicated across
    devices and receives every member's transitions. Otherwise each member has
    its own ring, sharded along the population axis. Replay stays at the workflow
    root so copying a parent during exploitation preserves the recipient's data.
    """

    def __init__(
        self, workflow: OffPolicyWorkflow, evaluator: Evaluator, config: DictConfig
    ):
        super().__init__(workflow, evaluator, config)
        self.replay_buffer = workflow.replay_buffer

    def setup(self, key: jax.Array):
        key, rb_key = jax.random.split(key)
        state = super().setup(key)

        state = state.replace(
            replay_buffer_state=self._setup_replaybuffer(rb_key),
        )

        logger.info("Start replay buffer post-setup")
        state = self._postsetup_replaybuffer(state)

        logger.info("Complete replay buffer post-setup")

        return state

    def _setup_replaybuffer(self, key: jax.Array) -> ReplayBufferState:
        if self.config.shared_replay_buffer:
            # Each device holds a replica of the shared replay buffer.
            return jax.device_put(self.workflow._setup_replaybuffer(key), jax.P())
        keys = jax.device_put(
            jax.random.split(key, self.config.pop_size), jax.P(META_POP_AXIS_NAME)
        )
        return shmap_vmap(
            self.workflow._setup_replaybuffer,
            in_specs=jax.P(META_POP_AXIS_NAME),
            out_specs=jax.P(META_POP_AXIS_NAME),
        )(keys)

    def _postsetup_replaybuffer(self, state: State) -> State:
        env = self.workflow.env
        num_envs = self.config.target_workflow.num_envs

        pop_workflow_state = state.pop_workflow_state
        replay_buffer_state = state.replay_buffer_state

        rollout_length = self.config.random_timesteps // num_envs
        if rollout_length < 1:
            raise ValueError(
                "random_timesteps must be at least target_workflow.num_envs"
            )

        # ==== fill random transitions ====

        key, rollout_key = jax.random.split(state.key)

        random_agent = RandomAgent()

        def random_trajectory(key):
            random_agent_state = random_agent.init(
                env.obs_space, env.action_space, jax.random.key(0)
            )
            env_key, rollout_key = jax.random.split(key)
            trajectory, _ = rollout(
                env_fn=env.step,
                action_fn=random_agent.compute_actions,
                env_state=env.reset(env_key),
                agent_state=random_agent_state,
                key=rollout_key,
                rollout_length=rollout_length,
                env_extra_fields=("ori_obs", "termination"),
            )
            # [T, B, ...] -> [T*B, ...]
            return tree_stop_gradient(
                flatten_rollout_trajectory(clean_trajectory(trajectory))
            )

        if self.config.shared_replay_buffer:
            trajectory = jax.device_put(random_trajectory(rollout_key), jax.P())
            replay_buffer_state = self.replay_buffer.add(
                replay_buffer_state, trajectory
            )
        else:
            keys = jax.device_put(
                jax.random.split(rollout_key, self.config.pop_size),
                jax.P(META_POP_AXIS_NAME),
            )

            def warmup_member(replay, key):
                trajectory = random_trajectory(key)
                return self.replay_buffer.add(replay, trajectory), trajectory.obs

            replay_buffer_state, observations = shmap_vmap(
                warmup_member,
                in_specs=jax.P(META_POP_AXIS_NAME),
                out_specs=jax.P(META_POP_AXIS_NAME),
            )(replay_buffer_state, keys)

        if pop_workflow_state.agent_state.obs_preprocessor_state is not None:
            obs_preprocessor_state = (
                pop_workflow_state.agent_state.obs_preprocessor_state
            )
            if self.config.shared_replay_buffer:
                obs_preprocessor_state = shmap_vmap(
                    running_statistics.update,
                    in_specs=(jax.P(META_POP_AXIS_NAME), jax.P()),
                    out_specs=jax.P(META_POP_AXIS_NAME),
                    vmap_in_axes=(0, None),
                )(obs_preprocessor_state, trajectory.obs)
            else:
                obs_preprocessor_state = shmap_vmap(
                    running_statistics.update,
                    in_specs=jax.P(META_POP_AXIS_NAME),
                    out_specs=jax.P(META_POP_AXIS_NAME),
                )(obs_preprocessor_state, observations)

            pop_workflow_state = pop_workflow_state.replace(
                agent_state=pop_workflow_state.agent_state.replace(
                    obs_preprocessor_state=obs_preprocessor_state
                )
            )

        sampled_timesteps_m = rollout_length * num_envs / 1e6
        if not self.config.shared_replay_buffer:
            sampled_timesteps_m *= self.config.pop_size
        workflow_metrics = state.metrics.replace(
            sampled_timesteps_m=state.metrics.sampled_timesteps_m + sampled_timesteps_m,
        )

        def reset_member(member):
            member_key, env_key = jax.random.split(member.key)
            return member.replace(key=member_key, env_state=env.reset(env_key))

        # Shared prefill uses one batch; restore the population's training
        # batches as well as resetting the actual host simulators.
        pop_workflow_state = shmap_vmap(
            reset_member,
            in_specs=jax.P(META_POP_AXIS_NAME),
            out_specs=jax.P(META_POP_AXIS_NAME),
        )(pop_workflow_state)

        return state.replace(
            key=key,
            metrics=workflow_metrics,
            pop_workflow_state=pop_workflow_state,
            replay_buffer_state=replay_buffer_state,
        )

    def step(self, state: State) -> tuple[MetricBase, State]:
        pop_workflow_state = state.pop_workflow_state
        pop = state.pop
        replay_buffer_state = state.replay_buffer_state
        pbt_opt_state = state.pbt_opt_state

        # ===== step ======
        def _train_steps(pop_wf_state, replay_buffer_state):
            def _one_step(carry, _):
                pop_wf_state, replay_buffer_state = carry

                def _wf_step_wrapper(wf_state, replay):
                    wf_state = wf_state.replace(replay_buffer_state=replay)
                    train_metrics, wf_state = self.workflow.step(wf_state)
                    wf_state = wf_state.replace(replay_buffer_state=None)
                    return train_metrics, wf_state

                pop_train_metrics, pop_wf_state = jax.vmap(
                    _wf_step_wrapper,
                    in_axes=(0, None if self.config.shared_replay_buffer else 0),
                )(pop_wf_state, replay_buffer_state)

                if self.config.shared_replay_buffer:
                    # [pop, T*B, ...] -> [pop*T*B, ...]
                    trajectory = jtu.tree_map(
                        lambda x: jax.lax.collapse(x, 0, 2),
                        pop_train_metrics.trajectory,
                    )
                    trajectory = all_gather(
                        trajectory,
                        META_POP_AXIS_NAME,
                        axis=0,
                        tiled=True,
                        to="invarying",
                    )
                    replay_buffer_state = self.replay_buffer.add(
                        replay_buffer_state, trajectory
                    )
                else:
                    replay_buffer_state = jax.vmap(self.replay_buffer.add)(
                        replay_buffer_state, pop_train_metrics.trajectory
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

        replay_spec = (
            jax.P() if self.config.shared_replay_buffer else jax.P(META_POP_AXIS_NAME)
        )
        pop_train_metrics, pop_workflow_state, replay_buffer_state = jax.shard_map(
            _train_steps,
            in_specs=(jax.P(META_POP_AXIS_NAME), replay_spec),
            out_specs=(
                jax.P(META_POP_AXIS_NAME),
                jax.P(META_POP_AXIS_NAME),
                replay_spec,
            ),
        )(pop_workflow_state, replay_buffer_state)

        # Count new interactions before PBT copies parent states into offspring.
        sampled_timesteps_m = (
            jnp.sum(
                (
                    pop_workflow_state.metrics.sampled_timesteps
                    - state.pop_workflow_state.metrics.sampled_timesteps
                ).astype(jnp.float32)
            )
            / 1e6
        )

        # ===== eval ======
        eval_fn = shmap_vmap(
            self.workflow.evaluate,
            in_specs=jax.P(META_POP_AXIS_NAME),
            out_specs=jax.P(META_POP_AXIS_NAME),
        )

        pop_eval_metrics, pop_workflow_state = eval_fn(pop_workflow_state)

        # customize your pop metrics here
        pop_episode_returns = pop_eval_metrics.episode_return

        # ===== warmup or exploit & explore ======
        key, exploit_and_explore_key = jax.random.split(state.key)

        def _dummy_fn(pbt_opt_state, pop, pop_workflow_state, pop_metrics, key):
            return pop, pop_workflow_state, pbt_opt_state

        pop, pop_workflow_state, pbt_opt_state = jax.lax.cond(
            state.metrics.iterations + 1
            <= math.ceil(
                self.config.warmup_steps / self.config.workflow_steps_per_iter
            ),
            _dummy_fn,
            self.exploit_and_explore,
            pbt_opt_state,
            pop,
            pop_workflow_state,
            pop_episode_returns,
            exploit_and_explore_key,
        )

        # ===== record metrics ======
        workflow_metrics = state.metrics.replace(
            sampled_timesteps_m=state.metrics.sampled_timesteps_m + sampled_timesteps_m,
            iterations=state.metrics.iterations + 1,
        )

        train_metrics = PBTOffpolicyTrainMetric(
            pop_episode_return=pop_eval_metrics.episode_return,
            pop_episode_length=pop_eval_metrics.episode_length,
            pop_train_metrics=pop_train_metrics,
            pop=state.pop,  # save prev pop instead of new pop to match the metrics
            rb_size=replay_buffer_state.buffer_size,
        )

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            pop=pop,
            pop_workflow_state=pop_workflow_state,
            pbt_opt_state=pbt_opt_state,
            replay_buffer_state=replay_buffer_state,
        )

    def _record_step_metrics(self, train_metrics, workflow_metrics, iters):
        train_metrics_dict = train_metrics.to_local_dict()

        train_metrics_dict["pop_episode_return"] = get_1d_array_statistics(
            train_metrics_dict["pop_episode_return"], histogram=True
        )
        train_metrics_dict["pop_episode_length"] = get_1d_array_statistics(
            train_metrics_dict["pop_episode_length"], histogram=True
        )
        train_metrics_dict["pop"] = convert_pop_to_df(train_metrics_dict["pop"])
        train_metrics_dict["pop_train_metrics"] = jtu.tree_map(
            get_1d_array_statistics, train_metrics_dict["pop_train_metrics"]
        )

        self.recorder.write(workflow_metrics.to_local_dict(), iters)
        self.recorder.write(train_metrics_dict, iters)

    def learn(self, state: State) -> State:
        for i in range(state.metrics.iterations, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            self._record_step_metrics(train_metrics, workflow_metrics, iters)

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
                saved_state,
                force=iters == self.config.num_iters,
            )

        return state

    @classmethod
    def enable_jit(cls) -> None:
        super().enable_jit()
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )
