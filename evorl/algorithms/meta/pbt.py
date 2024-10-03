import copy
import logging
import math
from functools import partial
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict, read_write

import chex
import hydra
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from optax.schedules import InjectStatefulHyperparamsState
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


from evorl.distributed import (
    POP_AXIS_NAME,
    parallel_map,
    tree_device_get,
    tree_device_put,
)
from evorl.metrics import MetricBase
from evorl.types import MISSING_REWARD, PyTreeDict, State
from evorl.utils.jax_utils import tree_last, tree_get, tree_set, is_jitted
from evorl.workflows import RLWorkflow, Workflow
from evorl.recorders import get_1d_array_statistics

from .pbt_operations import explore, select

logger = logging.getLogger(__name__)


class TrainMetric(MetricBase):
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array
    pop_train_metrics: MetricBase
    pop: chex.ArrayTree


class EvalMetric(MetricBase):
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array


class PBTWorkflowMetric(MetricBase):
    # the average of sampled timesteps of all workflows
    sampled_timesteps_m: chex.Array = jnp.zeros((), dtype=jnp.float32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class PBTWorkflow(Workflow):
    def __init__(self, workflow: RLWorkflow, config: DictConfig):
        super().__init__(config)

        self.workflow = workflow
        self.devices = jax.local_devices()[:1]
        self.sharding = None  # training sharding
        # self.pbt_update_sharding = None

    @classmethod
    def name(cls):
        return "PBT"

    @staticmethod
    def _rescale_config(config) -> None:
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

    def setup(self, key: chex.PRNGKey):
        pop_size = self.config.pop_size

        key, workflow_key, pop_key = jax.random.split(key, num=3)
        self.workflow.optimizer = optax.inject_hyperparams(
            optax.adam, static_args=("b1", "b2", "eps", "eps_root")
        )(learning_rate=self.config.search_space.lr.low)

        pop = PyTreeDict(
            lr=jax.random.uniform(
                pop_key,
                (pop_size,),
                minval=self.config.search_space.lr.low,
                maxval=self.config.search_space.lr.high,
            )
        )
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
        )(pop, pop_workflow_state)

        return State(
            key=key,
            metrics=workflow_metrics,
            pop_workflow_state=pop_workflow_state,
            pop=pop,
        )

    def step(self, state: State) -> tuple[TrainMetric, State]:
        pop_workflow_state = state.pop_workflow_state
        pop = state.pop

        # ===== step ======
        def _train_steps(wf_state):
            def _one_step(wf_state, _):
                train_metrics, wf_state = self.workflow.step(wf_state)
                return wf_state, train_metrics

            wf_state, train_metrics_list = jax.lax.scan(
                _one_step, wf_state, (), length=self.config.workflow_steps_per_iter
            )

            train_metrics = tree_last(train_metrics_list)

            return train_metrics, wf_state

        if self.config.parallel_train:
            train_steps_fn = jax.vmap(_train_steps, spmd_axis_name=POP_AXIS_NAME)
        else:
            # TODO: fix potential unneccesary gpu-comm: eg: all-gather in ppo #line=387
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

        # _exploit_and_explore_fn = shard_map(
        #     self.exploit_and_explore,
        #     mesh=self.pbt_update_sharding.mesh,
        #     in_specs=self.pbt_update_sharding.spec,
        #     out_specs=self.pbt_update_sharding.spec,
        # )
        _exploit_and_explore_fn = self.exploit_and_explore

        pop, pop_workflow_state = jax.lax.cond(
            state.metrics.iterations + 1
            <= math.ceil(
                self.config.warmup_steps / self.config.workflow_steps_per_iter
            ),
            _dummy_fn,
            _exploit_and_explore_fn,
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

        train_metrics = TrainMetric(
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

    def learn(self, state: State) -> State:
        for i in range(state.metrics.iterations, self.config.num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            self.recorder.write(workflow_metrics.to_local_dict(), i)

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

            train_metrics_dict["pop"] = _convert_pop_to_df(train_metrics_dict["pop"])

            train_metrics_dict["pop_train_metrics"] = jtu.tree_map(
                get_1d_array_statistics, train_metrics_dict["pop_train_metrics"]
            )

            self.recorder.write(train_metrics_dict, i)

            self.checkpoint_manager.save(i, args=ocp.args.StandardSave(state))

        return state

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
        eval_metrics = EvalMetric(
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
    ):
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

        # Note: no need to deepcopy parents wf_state here
        offsprings_workflow_state = jax.vmap(self.apply_hyperparams_to_workflow_state)(
            offsprings, parents_wf_state
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

    def apply_hyperparams_to_workflow_state(
        self, hyperparams: PyTreeDict[str, chex.Numeric], workflow_state: State
    ):
        """
        Note1: InjectStatefulHyperparamsState is NamedTuple, which is not immutable.
        Note2: try to avoid deepcopy unnessary state
        """
        opt_state = workflow_state.opt_state
        assert isinstance(opt_state, InjectStatefulHyperparamsState)

        opt_state = deepcopy_InjectStatefulHyperparamsState(opt_state)
        opt_state.hyperparams["learning_rate"] = hyperparams.lr
        return workflow_state.replace(opt_state=opt_state)

    @classmethod
    def enable_jit(cls) -> None:
        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls.step = jax.jit(cls.step, static_argnums=(0,))


def _pop_write(indices, pop, new):
    return jtu.tree_map(lambda x, y: x.at[indices].set(y), pop, new)


def _convert_pop_to_df(pop):
    df = pd.DataFrame.from_dict(pop)
    df.insert(0, "pop_id", range(len(df)))
    return df


def deepcopy_InjectStatefulHyperparamsState(state: InjectStatefulHyperparamsState):
    return InjectStatefulHyperparamsState(
        count=state.count,
        hyperparams=copy.deepcopy(state.hyperparams),
        hyperparams_states=state.hyperparams_states,
        inner_state=state.inner_state,
    )
