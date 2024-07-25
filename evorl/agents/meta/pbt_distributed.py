import copy
import logging
import math
from functools import partial

import chex
import hydra
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import DictConfig, OmegaConf, open_dict, read_write
from optax.schedules import InjectStatefulHyperparamsState

from evorl.distributed import POP_AXIS_NAME, tree_device_get, tree_device_put
from evorl.metrics import MetricBase, TrainMetric
from evorl.types import PyTreeData, PyTreeDict, State
from evorl.utils.jax_utils import tree_last
from evorl.workflows import RLWorkflow

logger = logging.getLogger(__name__)


class TrainMetric(MetricBase):
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array
    pop_train_metrics: MetricBase


class EvalMetric(MetricBase):
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array


class HyperParams(PyTreeData):
    lr: chex.Array


class PBTWorkflow(RLWorkflow):
    def __init__(self, workflow: RLWorkflow, config: DictConfig):
        super().__init__(config)

        self.workflow = workflow

        self.sharding = None

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
        cls._rescale_config(config, devices)

        if enable_jit:
            cls.enable_jit()

        OmegaConf.set_readonly(config, True)

        workflow = cls._build_from_config(config)

        workflow.pmap_axis_name = None  # we use shmap instead.
        workflow.devices = devices

        mesh = Mesh(devices, axis_names=(POP_AXIS_NAME,))
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

        target_workflow = target_workflow_cls.build_from_config(
            target_workflow_config, enable_jit=True
        )

        target_workflow.devices = devices

        return cls(target_workflow, config)

    def setup(self, key: chex.PRNGKey):
        self.recorder.init()

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
        workflow_metrics = self._setup_workflow_metrics()

        workflow_keys = jax.random.split(workflow_key, pop_size)
        workflow_keys = jax.device_put(workflow_keys, self.sharding)

        pop_workflow_state = jax.jit(
            jax.vmap(self.workflow.setup, spmd_axis_name=POP_AXIS_NAME),
            in_shardings=self.sharding,
            out_shardings=self.sharding,
        )(workflow_keys)

        pop_workflow_state = jax.jit(
            jax.vmap(apply_hyperparams_to_workflow_state, spmd_axis_name=POP_AXIS_NAME),
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

            wf_state, train_metrics_trajectory = jax.lax.scan(
                _one_step, wf_state, None, length=self.config.per_iter_workflow_steps
            )

            # jax.debug.print("{x}", x=train_metrics_trajectory.train_episode_return)

            train_metrics = tree_last(train_metrics_trajectory)

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

        pop_episode_returns = pop_eval_metrics.episode_returns

        # ===== warmup or exploit & explore ======
        key, exploit_and_explore_key = jax.random.split(state.key)

        def _dummy_fn(key, pop_episode_returns, pop, pop_workflow_state):
            return pop, pop_workflow_state

        _exploit_and_explore_fn = partial(exploit_and_explore, self.config)

        pop, pop_workflow_state = jax.lax.cond(
            state.metrics.iterations + 1
            <= math.ceil(
                self.config.warmup_steps / self.config.per_iter_workflow_steps
            ),
            _dummy_fn,
            _exploit_and_explore_fn,
            exploit_and_explore_key,
            pop_episode_returns,
            pop,
            pop_workflow_state,
        )

        # ===== record metrics ======
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=jnp.sum(pop_workflow_state.metrics.sampled_timesteps),
            iterations=state.metrics.iterations + 1,
        )

        train_metrics = TrainMetric(
            pop_episode_returns=pop_eval_metrics.episode_returns,
            pop_episode_lengths=pop_eval_metrics.episode_lengths,
            pop_train_metrics=pop_train_metrics,
        )

        train_metrics = tree_device_get(train_metrics, self.devices[0])

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            pop=pop,
            pop_workflow_state=pop_workflow_state,
        )

    def learn(self, state: State) -> State:
        for i in range(self.config.num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics
            self.recorder.write(workflow_metrics.to_local_dict(), i)
            self.recorder.write(train_metrics.to_local_dict(), i)

            self.checkpoint_manager.save(i, args=ocp.args.StandardSave(state))

        return state

    def evaluate(self, state: State) -> State:
        # Tips: evaluation consumes every workflow_state's internal key
        if self.config.parallel_eval:
            pop_eval_metrics, pop_workflow_state = jax.vmap(self.workflow.evaluate)(
                pop_workflow_state
            )
        else:
            pop_eval_metrics, pop_workflow_state = jax.lax.map(
                self.workflow.evaluate, pop_workflow_state
            )

        eval_metrics = EvalMetric(
            pop_episode_returns=pop_eval_metrics.episode_returns,
            pop_episode_lengths=pop_eval_metrics.episode_lengths,
        )

        return eval_metrics, state.replace(pop_workflow_state=pop_workflow_state)


def exploit_and_explore(config, key, pop_episode_returns, pop, pop_workflow_state):
    bottoms_num = round(config.pop_size * config.bottom_ratio)
    tops_num = round(config.pop_size * config.top_ratio)
    indices = jnp.argsort(pop_episode_returns)
    bottoms_indices = indices[:bottoms_num]
    tops_indices = indices[-tops_num:]

    key, exploit_key, explore_key = jax.random.split(key, 3)

    def _exploit_and_explore_fn(key, top, top_wf_state):
        new = PyTreeDict()
        for hp_name in top.keys():
            new[hp_name] = top[hp_name] * (
                1
                + jax.random.uniform(
                    key,
                    minval=-config.perturb_factor[hp_name],
                    maxval=config.perturb_factor[hp_name],
                )
            )
        # TODO: check deepcopy is necessary (does not change the original state)

        new_wf_state = apply_hyperparams_to_workflow_state(new, top_wf_state)
        return new, new_wf_state

    # replace bottoms with random tops
    tops_choice_indices = jax.random.choice(
        exploit_key, tops_indices, (len(bottoms_indices),)
    )

    def _read(indices, pop):
        return jtu.tree_map(lambda x: x[indices], pop)

    def _write(indices, pop, new):
        return jtu.tree_map(lambda x, y: x.at[indices].set(y), pop, new)

    new_bottoms, new_bottoms_wf_state = jax.vmap(_exploit_and_explore_fn)(
        jax.random.split(explore_key, len(bottoms_indices)),
        _read(tops_choice_indices, pop),
        _read(tops_choice_indices, pop_workflow_state),
    )
    pop = _write(bottoms_indices, pop, new_bottoms)
    pop_workflow_state = _write(
        bottoms_indices, pop_workflow_state, new_bottoms_wf_state
    )

    return pop, pop_workflow_state


def apply_hyperparams_to_workflow_state(
    hyperparams: PyTreeDict[str, chex.Numeric], workflow_state: State
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


def deepcopy_InjectStatefulHyperparamsState(state: InjectStatefulHyperparamsState):
    return InjectStatefulHyperparamsState(
        count=state.count,
        hyperparams=copy.deepcopy(state.hyperparams),
        hyperparams_states=state.hyperparams_states,
        inner_state=state.inner_state,
    )


def parallel_map(fn, sharding):
    """
    sequential on the same gpu, parrallel on different gpu.
    """

    def shmap_f(state):
        # state: sharded state on single device
        # jax.debug.print("{}", state.env_state.obs.shape)
        return jax.lax.map(fn, state)

    return shard_map(
        shmap_f,
        mesh=sharding.mesh,
        in_specs=sharding.spec,
        out_specs=sharding.spec,
        check_rep=False,
    )
