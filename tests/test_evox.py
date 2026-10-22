"""Optional PyTorch EvoX integration: state isolation, DLPack and RL evaluation."""

from pathlib import Path
import weakref
from unittest.mock import Mock, patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf

torch = pytest.importorskip("torch", minversion="2.9.0")
pytest.importorskip("evox")

from evox.algorithms import CSO, NSGA2, RVEA as EvoXRVEA
from evox.core import Algorithm, Mutable, Parameter
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    DataDependentOutputException,
)

from evorl.algorithms.contrib.evox.evox_algorithm_wrapper import (
    EvoXAlgorithmWrapper,
)
from evorl.algorithms.contrib.evox.evox_workflow import EvoXWorkflow
from evorl.algorithms.contrib.evox.rvea import RVEA
from evorl.algorithms.ec.obs_utils import init_obs_preprocessor
from evorl.utils.orbax_utils import CheckpointManager, load, save
from evorl.utils.torch_utils import fork_torch_rng, jax_to_torch_device
from evorl.recorders import LogRecorder
from evorl.utils.orbax_utils import DummyCheckpointManager
from evorl.workflows import Workflow
from evorl.types import PyTreeDict
from tests.helpers.envs import ToyEnv


def torch_device():
    return jax_to_torch_device(jax.local_devices()[0])


@jax.jit
def evaluate_rewards(population, evaluation_state):
    return -jnp.square(population).sum(-1), (), evaluation_state


def make_wrapper(
    algorithm_cls,
    algorithm_kwargs,
    *,
    device,
    compile=False,
    evaluate=evaluate_rewards,
    static_tensors=None,
):
    return EvoXAlgorithmWrapper(
        algorithm_cls,
        algorithm_kwargs,
        device=device,
        compile=compile,
        evaluate=evaluate,
        static_tensors=static_tensors,
    )


def rng_states():
    states = [torch.random.get_rng_state().clone()]
    device = torch_device()
    if device.type != "cpu":
        states.append(
            torch.get_device_module(device).get_rng_state(device.index).clone()
        )
    return states


def assert_rng_states_equal(expected):
    assert all(torch.equal(left, right) for left, right in zip(expected, rng_states()))


class StatefulAlgorithm(Algorithm):
    """Exercise registered parameters, non-persistent buffers and in-place writes."""

    def __init__(self, device):
        super().__init__()
        self.rate = Parameter(0.5, device=device)
        self.pop = Mutable(torch.randn(4, 2, device=device))
        self.fit = Mutable(torch.zeros(4, device=device))
        self.register_buffer(
            "counter",
            torch.tensor(0, dtype=torch.int32, device=device),
            persistent=False,
        )

    def init_step(self):
        self.fit = self.evaluate(self.pop)

    def step(self):
        # Exercise both default generators when using an accelerator.
        cpu_noise = torch.rand((), device="cpu").to(self.pop.device)
        self.pop.add_(self.rate * torch.randn_like(self.pop) + cpu_noise)
        self.counter.add_(1)
        self.fit.copy_(self.evaluate(self.pop))


class MutatingConstructorAlgorithm(StatefulAlgorithm):
    """Use and mutate tensor/container arguments while constructing an algorithm."""

    def __init__(self, device, initial_population, options):
        super().__init__(device)
        self.pop = Mutable(initial_population)
        self.pop.add_(options["offsets"].pop())


def test_constructor_arguments_are_isolated_across_callers_and_initializations():
    device = torch_device()
    population = torch.ones(4, 2, device=device)
    options = dict(offsets=[0.25])
    kwargs = dict(device=device, initial_population=population, options=options)
    before = rng_states()
    wrapper = make_wrapper(MutatingConstructorAlgorithm, kwargs, device=device)
    assert options == dict(offsets=[0.25])
    torch.testing.assert_close(population, torch.ones_like(population))
    assert_rng_states_equal(before)

    # Changing caller-owned arguments must not change the saved configuration.
    population.add_(100)
    options["offsets"][0] = 7.0
    kwargs["initial_population"] = torch.zeros_like(population)
    _, first = wrapper.init(jax.random.key(17))
    _, second = wrapper.init(jax.random.key(17))
    chex.assert_trees_all_equal(first, second)
    np.testing.assert_array_equal(
        first.params_and_buffers["pop"], np.full((4, 2), 1.25)
    )
    assert options == dict(offsets=[7.0])
    torch.testing.assert_close(population, torch.full_like(population, 101))
    assert_rng_states_equal(before)


def test_constructor_failure_restores_rng():
    class FailingAlgorithm(StatefulAlgorithm):
        def __init__(self, device):
            super().__init__(device)
            raise RuntimeError("construction failed")

    device = torch_device()
    before = rng_states()
    with pytest.raises(RuntimeError, match="construction failed"):
        make_wrapper(FailingAlgorithm, dict(device=device), device=device)
    assert_rng_states_equal(before)


@pytest.mark.parametrize(
    "dtype", [torch.int64, torch.uint64, torch.float64, torch.complex128]
)
def test_64bit_state_is_rejected_without_changing_rng(dtype):
    class WideBufferAlgorithm(StatefulAlgorithm):
        def __init__(self, device):
            super().__init__(device)
            self.wide = Mutable(torch.ones(1, dtype=dtype, device=device))

    device = torch_device()
    before = rng_states()
    with pytest.raises(TypeError, match="'wide'.*64-bit.*adjust the EvoX algorithm"):
        make_wrapper(WideBufferAlgorithm, dict(device=device), device=device)
    assert_rng_states_equal(before)


def test_64bit_state_created_during_step_is_rejected():
    class WideUpdateAlgorithm(StatefulAlgorithm):
        def step(self):
            super().step()
            self.counter = self.counter.to(torch.int64)

    device = torch_device()
    wrapper = make_wrapper(WideUpdateAlgorithm, dict(device=device), device=device)
    _, state = wrapper.init(jax.random.key(9))
    before = rng_states()
    with pytest.raises(TypeError, match="'counter'.*64-bit"):
        wrapper.step(state)
    assert state.params_and_buffers["counter"].dtype == jnp.int32
    assert wrapper._evaluation_state is None
    assert wrapper._evaluation_metrics is None
    assert_rng_states_equal(before)


def test_64bit_input_state_is_rejected_before_running_algorithm():
    device = torch_device()
    wrapper = make_wrapper(StatefulAlgorithm, dict(device=device), device=device)
    _, state = wrapper.init(jax.random.key(9))
    with jax.enable_x64():
        wide_counter = state.params_and_buffers["counter"].astype(jnp.int64)
    state = state.replace(
        params_and_buffers=dict(state.params_and_buffers, counter=wide_counter)
    )
    with pytest.raises(TypeError, match="'counter'.*64-bit"):
        wrapper.step(state)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_dlpack_evaluation_keeps_device_and_dtype(dtype):
    device = torch_device()
    problem = make_wrapper(
        StatefulAlgorithm,
        dict(device=device),
        device=device,
        evaluate=jax.jit(lambda pop, state: (pop.sum(-1).astype(dtype), (), state)),
    )
    population = torch.ones(4, 3, device=device)
    result = problem._evaluate(population)
    assert result.device == population.device
    assert result.dtype == (torch.float32 if dtype == jnp.float32 else torch.bfloat16)
    torch.testing.assert_close(
        result, torch.full((4,), -3.0, device=device).to(result.dtype)
    )


def test_evaluation_rejects_64bit_population_and_fitness():
    problem = make_wrapper(
        StatefulAlgorithm,
        dict(device=torch_device()),
        device=torch_device(),
        evaluate=Mock(),
    )
    with pytest.raises(TypeError, match="Population.*64-bit"):
        problem._evaluate(torch.ones(4, 3, dtype=torch.float64, device=torch_device()))
    problem.evaluate_fn.assert_not_called()
    with jax.enable_x64():
        problem = make_wrapper(
            StatefulAlgorithm,
            dict(device=torch_device()),
            device=torch_device(),
            evaluate=jax.jit(
                lambda population, state: (
                    population.sum(-1).astype(jnp.float64),
                    (),
                    state,
                )
            ),
        )
        with pytest.raises(TypeError, match="JAX fitness.*64-bit"):
            problem._evaluate(torch.ones(4, 3, device=torch_device()))


def test_fake_population_cannot_reach_dlpack_evaluation():
    problem = make_wrapper(
        StatefulAlgorithm,
        dict(device=torch_device()),
        device=torch_device(),
        evaluate=Mock(),
    )
    with FakeTensorMode():
        population = torch.ones(4, 3, device=torch_device())
    with pytest.raises(TypeError, match="FakeTensor"):
        problem._evaluate(population)
    problem.evaluate_fn.assert_not_called()


def test_unregistered_fake_constants_must_be_provided_explicitly():
    device = torch_device()
    with pytest.raises(ValueError, match="'lb'.*static_tensors"):
        make_wrapper(
            CSO,
            dict(
                pop_size=4,
                lb=-torch.ones(3, device=device),
                ub=torch.ones(3, device=device),
                device=device,
            ),
            device=device,
        )


def test_static_constants_are_real_copied_and_unchanged_by_init(jit):
    device = torch_device()
    lb, ub = -torch.ones(3, device=device), torch.ones(3, device=device)
    bounds = dict(lb=lb[None, :], ub=ub[None, :])
    wrapper = make_wrapper(
        CSO,
        dict(pop_size=4, lb=lb, ub=ub, device=device),
        device=device,
        compile=jit,
        static_tensors=bounds,
    )
    assert isinstance(wrapper._algorithm.pop, FakeTensor)
    assert not isinstance(wrapper._algorithm.lb, FakeTensor)
    constants = wrapper._algorithm.lb, wrapper._algorithm.ub
    lb.fill_(-100)
    ub.fill_(100)
    bounds.clear()
    _, state = wrapper.init(jax.random.key(9))
    assert (state.params_and_buffers["pop"] >= -1).all()
    assert (state.params_and_buffers["pop"] <= 1).all()
    for _ in range(2):
        _, state = wrapper.step(state)
        assert not isinstance(state.params_and_buffers["pop"], FakeTensor)
    assert wrapper._algorithm.lb is constants[0]
    assert wrapper._algorithm.ub is constants[1]
    torch.testing.assert_close(constants[0], -torch.ones(1, 3, device=device))
    torch.testing.assert_close(constants[1], torch.ones(1, 3, device=device))


@pytest.mark.parametrize(
    "wrong_output", ["scalar", "candidates", "rank", "dtype", "empty"]
)
def test_evaluation_rejects_misaligned_fitness(wrong_output):
    def evaluate(population, state):
        fitness = population.sum(-1)
        return (
            {
                "scalar": fitness.sum(),
                "candidates": fitness[:-1],
                "rank": fitness[:, None, None],
                "dtype": fitness.astype(jnp.int32),
                "empty": fitness[:, None][:, :0],
            }[wrong_output],
            (),
            state,
        )

    problem = make_wrapper(
        StatefulAlgorithm,
        dict(device=torch_device()),
        device=torch_device(),
        evaluate=jax.jit(evaluate),
    )
    with pytest.raises(ValueError, match="Fitness must be a floating JAX array"):
        problem._evaluate(torch.ones(4, 3, device=torch_device()))


def test_constructor_prepares_execution_and_init_preserves_fixed_components(jit):
    class CountingAlgorithm(StatefulAlgorithm):
        constructions = 0

        def __init__(self, device):
            super().__init__(device)
            type(self).constructions += 1

    device = torch_device()
    before = rng_states()
    wrapper = make_wrapper(
        CountingAlgorithm, dict(device=device), device=device, compile=jit
    )
    assert CountingAlgorithm.constructions == 1
    assert isinstance(wrapper._algorithm.pop, FakeTensor)
    fixed_attributes = vars(wrapper).copy()
    _, first = wrapper.init(jax.random.key(17))
    assert CountingAlgorithm.constructions == 2
    assert all(vars(wrapper)[name] is value for name, value in fixed_attributes.items())
    cached = (wrapper._algorithm, wrapper._init_step, wrapper._step)
    expected = wrapper.step(first)[1]
    _, second = wrapper.init(jax.random.key(17))
    _, third = wrapper.init(jax.random.key(18))
    assert CountingAlgorithm.constructions == 4
    assert all(vars(wrapper)[name] is value for name, value in fixed_attributes.items())
    assert all(
        left is right
        for left, right in zip(
            cached, (wrapper._algorithm, wrapper._init_step, wrapper._step)
        )
    )
    chex.assert_trees_all_equal(first, second)
    assert not np.array_equal(
        first.params_and_buffers["pop"], third.params_and_buffers["pop"]
    )
    chex.assert_trees_all_equal(wrapper.step(first)[1], expected)
    assert_rng_states_equal(before)


def test_constructor_device_context_preserves_tied_parameters():
    class TiedAlgorithm(StatefulAlgorithm):
        def __init__(self, device):
            super().__init__(device)
            # Exercise factories that omit device, as some EvoX algorithms do.
            self.rate = Parameter(0.5)
            self.tied_rate = self.rate
            self.offset = torch.ones(2)

    device = torch_device()
    wrapper = make_wrapper(
        TiedAlgorithm,
        dict(device=device),
        device=device,
        static_tensors=dict(offset=torch.ones(2, device=device)),
    )
    assert wrapper._algorithm.rate is wrapper._algorithm.tied_rate
    assert wrapper._algorithm.rate.device == device
    assert wrapper._algorithm.offset.device == device
    with patch.object(
        TiedAlgorithm, "to", side_effect=AssertionError("unexpected migration")
    ):
        _, state = wrapper.init(jax.random.key(9))
    assert all(
        value.device == jax.local_devices()[0]
        for value in state.params_and_buffers.values()
    )
    _, first = wrapper.step(state)
    chex.assert_trees_all_equal(first, wrapper.step(state)[1])
    assert wrapper._algorithm.rate is wrapper._algorithm.tied_rate


def test_explicitly_misplaced_algorithm_tensors_are_rejected():
    device = torch_device()
    if device.type == "cpu":
        pytest.skip("Requires distinct CPU and accelerator devices")

    class MisplacedAlgorithm(StatefulAlgorithm):
        def __init__(self, device):
            super().__init__(device)
            self.rate = Parameter(0.5, device="cpu")

    before = rng_states()
    with pytest.raises(ValueError, match="'rate'.*cpu.*construct it on"):
        make_wrapper(MisplacedAlgorithm, dict(device=device), device=device)
    assert_rng_states_equal(before)


def test_failed_constructor_preparation_preserves_rng():
    device = torch_device()
    before = rng_states()
    with (
        patch(
            "evorl.algorithms.contrib.evox.evox_algorithm_wrapper.use_state",
            side_effect=RuntimeError("preparation failed"),
        ),
        pytest.raises(RuntimeError, match="preparation failed"),
    ):
        make_wrapper(StatefulAlgorithm, dict(device=device), device=device)
    assert_rng_states_equal(before)
    wrapper = make_wrapper(StatefulAlgorithm, dict(device=device), device=device)
    _, state = wrapper.init(jax.random.key(17))
    chex.assert_trees_all_equal(state, wrapper.init(jax.random.key(17))[1])
    assert_rng_states_equal(before)


@pytest.mark.parametrize("compile", [False, True])
def test_init_runs_native_first_update_and_threads_explicit_evaluation_state(compile):
    class DefaultFirstStepAlgorithm(StatefulAlgorithm):
        init_step = Algorithm.init_step

    @jax.jit
    def evaluate(population, state):
        return (
            -jnp.square(population).sum(-1),
            population.shape[0],
            state.replace(calls=state.calls + 1),
        )

    device = torch_device()
    wrapper = make_wrapper(
        DefaultFirstStepAlgorithm,
        dict(device=device),
        device=device,
        compile=compile,
        evaluate=evaluate,
    )
    context = PyTreeDict(calls=jnp.array(0))
    before = rng_states()
    first_metrics, first = wrapper.init(jax.random.key(3), context)
    first_context = first.evaluation_state
    assert first_metrics == (4,)
    assert int(first.params_and_buffers["counter"]) == 1
    assert int(first_context.calls) == 1
    assert int(context.calls) == 0
    assert wrapper.evaluate_fn is evaluate
    assert wrapper._evaluation_state is None
    assert wrapper._evaluation_metrics is None
    second_metrics, second = wrapper.step(first)
    second_context = second.evaluation_state
    assert second_metrics == (4,)
    assert int(second.params_and_buffers["counter"]) == 2
    assert int(second_context.calls) == 2
    assert int(first_context.calls) == 1
    assert wrapper.evaluate_fn is evaluate
    assert wrapper._evaluation_state is None
    assert wrapper._evaluation_metrics is None
    assert_rng_states_equal(before)


def test_initial_evaluation_failure_restores_rng_and_does_not_retain_context():
    def evaluate(population, state):
        if state == "fail":
            raise RuntimeError("initial evaluation failed")
        return evaluate_rewards(population, None)[0], (), state

    device = torch_device()
    wrapper = make_wrapper(
        StatefulAlgorithm, dict(device=device), device=device, evaluate=evaluate
    )
    before = rng_states()
    with pytest.raises(RuntimeError, match="initial evaluation failed"):
        wrapper.init(jax.random.key(3), "fail")
    assert wrapper._evaluation_state is None
    assert wrapper._evaluation_metrics is None
    assert_rng_states_equal(before)
    _, first = wrapper.init(jax.random.key(3))
    _, second = wrapper.init(jax.random.key(3))
    chex.assert_trees_all_equal(first, second)
    assert_rng_states_equal(before)


def test_complete_state_isolated_replay_and_exception_restore(jit):
    def evaluate(population, evaluation_state):
        if evaluation_state == "fail":
            raise RuntimeError("evaluation failed")
        return evaluate_rewards(population, None)[0], (), evaluation_state

    device = torch_device()
    before = rng_states()
    wrapper = make_wrapper(
        StatefulAlgorithm,
        dict(device=device),
        device=device,
        compile=jit,
        evaluate=evaluate,
    )
    _, state = wrapper.init(jax.random.key(17))
    assert_rng_states_equal(before)
    assert set(state.params_and_buffers) == {"rate", "pop", "fit", "counter"}
    assert state.params_and_buffers["counter"].dtype == jnp.int32
    assert int(state.params_and_buffers["counter"]) == 0
    snapshot = jax.tree.map(lambda x: np.array(x, copy=True), state)
    placeholders = dict(wrapper._algorithm.named_parameters()) | dict(
        wrapper._algorithm.named_buffers()
    )
    _, first = wrapper.step(state)
    _, second = wrapper.step(state)
    chex.assert_trees_all_equal(first, second)
    chex.assert_trees_all_equal(state, snapshot)
    assert int(first.params_and_buffers["counter"]) == 1
    assert wrapper._evaluation_state is None
    assert wrapper._evaluation_metrics is None
    actual = dict(wrapper._algorithm.named_parameters()) | dict(
        wrapper._algorithm.named_buffers()
    )
    assert all(actual[name] is value for name, value in placeholders.items())
    with pytest.raises(RuntimeError, match="evaluation failed"):
        wrapper.step(state.replace(evaluation_state="fail"))
    assert wrapper._evaluation_state is None
    assert wrapper._evaluation_metrics is None
    assert_rng_states_equal(before)
    chex.assert_trees_all_equal(state, snapshot)
    chex.assert_trees_all_equal(wrapper.step(state)[1], first)

    class Payload:
        pass

    payload = Payload()
    reference = weakref.ref(payload)
    wrapper.step(state.replace(evaluation_state=payload))
    del payload
    assert reference() is None


@pytest.mark.parametrize("multi_objective", [False, True])
def test_real_algorithms_maximize_rewards_and_reuse_jitted_evaluation(
    multi_objective, jit
):
    device = torch_device()
    bounds = dict(
        lb=torch.full((3,), -1.0, device=device), ub=torch.ones(3, device=device)
    )
    cls = NSGA2 if multi_objective else CSO
    kwargs = {"n_objs": 2} if multi_objective else {}
    shapes = []

    @jax.jit
    def evaluate(population, evaluation_state):
        shapes.append(population.shape)
        fitness = -jnp.square(population).sum(-1)
        return (
            (jnp.stack((fitness, fitness - 1), -1) if multi_objective else fitness),
            (),
            evaluation_state,
        )

    wrapper = make_wrapper(
        cls,
        dict(pop_size=8, device=device, **bounds, **kwargs),
        device=device,
        compile=jit,
        evaluate=evaluate,
        static_tensors=dict(
            lb=bounds["lb"] if multi_objective else bounds["lb"][None, :],
            ub=bounds["ub"] if multi_objective else bounds["ub"][None, :],
        ),
    )
    _, state = wrapper.init(jax.random.key(12))
    for _ in range(3):
        snapshot = jax.tree.map(lambda x: np.array(x, copy=True), state)
        _, next_state = wrapper.step(state)
        chex.assert_trees_all_equal(state, snapshot)
        state = next_state
        pop = state.params_and_buffers["pop"]
        expected = -jnp.square(pop).sum(-1)
        if multi_objective:
            expected = jnp.stack((expected, expected - 1), -1)
        # EvoX stores minimizing fitness, so rewards must be negated once.
        np.testing.assert_allclose(
            state.params_and_buffers["fit"], -expected, atol=1e-6
        )
    assert shapes == ([(8, 3)] if multi_objective else [(8, 3), (4, 3)])


def test_state_checkpoint_roundtrip_reproduces_next_iteration(tmp_path):
    device = torch_device()
    wrapper = make_wrapper(StatefulAlgorithm, dict(device=device), device=device)
    _, state = wrapper.init(jax.random.key(8))
    _, state = wrapper.step(state)
    path = tmp_path / "evox"
    with CheckpointManager(path) as manager:
        manager.save(1, state, force=True)
        manager.wait_until_finished()
        restored = manager.restore(1, state)
    resumed_wrapper = make_wrapper(
        StatefulAlgorithm, dict(device=device), device=device
    )
    # A new wrapper can consume restored state without initializing another state.
    chex.assert_trees_all_equal(
        wrapper.step(state)[1], resumed_wrapper.step(restored)[1]
    )


def test_constructor_requiring_tensor_values_cannot_build_fake_template():
    class DataDependentAlgorithm(StatefulAlgorithm):
        def __init__(self, device):
            super().__init__(device)
            self.total = float(self.pop.sum())

    device = torch_device()
    with pytest.raises(DataDependentOutputException):
        make_wrapper(
            DataDependentAlgorithm,
            dict(device=device),
            device=device,
        )


class MultipleEvaluationAlgorithm(Algorithm):
    def __init__(self, pop_size, lb, ub, device):
        super().__init__()
        self.pop = Mutable(
            torch.rand(pop_size, lb.numel(), device=device) * (ub - lb) + lb
        )
        self.fit = Mutable(torch.zeros(pop_size, device=device))

    def step(self):
        self.evaluate(self.pop[:2])
        self.fit = self.evaluate(self.pop)


class NSGA2TestWorkflow(EvoXWorkflow):
    """Exercise shared vector-reward handling, including a size-one vector."""

    @classmethod
    def _build_algorithm(cls, config, *, param_dim, num_objectives, device, evaluate):
        return EvoXAlgorithmWrapper(
            NSGA2,
            dict(
                pop_size=config.pop_size,
                n_objs=num_objectives,
                lb=torch.full((param_dim,), config.agent_network.lb, device=device),
                ub=torch.full((param_dim,), config.agent_network.ub, device=device),
                device=device,
            ),
            device=device,
            compile=config.torch_compile,
            evaluate=evaluate,
        )


class MultipleEvaluationWorkflow(EvoXWorkflow):
    @classmethod
    def _build_algorithm(cls, config, *, param_dim, num_objectives, device, evaluate):
        return EvoXAlgorithmWrapper(
            MultipleEvaluationAlgorithm,
            dict(
                pop_size=config.pop_size,
                lb=torch.full((param_dim,), config.agent_network.lb, device=device),
                ub=torch.full((param_dim,), config.agent_network.ub, device=device),
                device=device,
            ),
            device=device,
            compile=config.torch_compile,
            evaluate=evaluate,
        )


class VectorRewardEnv(ToyEnv):
    def __init__(self, num_envs, objectives=2):
        super().__init__(num_envs)
        self.objectives = objectives

    def reset(self, key):
        return (
            super()
            .reset(key)
            .replace(reward=jnp.zeros((self.num_envs, self.objectives)))
        )

    def step(self, state, action):
        state = super().step(state, action)
        reward = jnp.stack(
            [1 - jnp.square(action - i * 0.5).sum(-1) for i in range(self.objectives)],
            axis=-1,
        )
        return state.replace(reward=reward)


def workflow_config(algorithm="cso"):
    root = Path(__file__).resolve().parents[1]
    return OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / f"configs/agent/contrib/{algorithm}-evox.yaml"),
        {
            "env": {"max_episode_steps": 2},
            "num_iters": 2,
            "pop_size": 4,
            "num_envs": 2,
            "episodes_for_fitness": 3,
            "normalize_obs": False,
            "agent_network": {"actor_hidden_layer_sizes": [2]},
        },
    )


def workflow_state_with_key_data(state):
    evaluation_state = state.algorithm_state.evaluation_state
    return state.replace(
        algorithm_state=state.algorithm_state.replace(
            evaluation_state=evaluation_state.replace(
                key=jax.random.key_data(evaluation_state.key)
            )
        )
    )


@pytest.mark.integration
@pytest.mark.parametrize("jax_jit", [False, True])
@pytest.mark.parametrize(
    "objectives", [None, 1, 2], ids=["scalar", "vector1", "vector2"]
)
def test_common_workflow_counts_actual_evaluations_and_resumes(
    objectives, jit, jax_jit, tmp_path
):
    config = workflow_config()
    config.num_iters = 1
    base = get_class(config.workflow_cls) if objectives is None else NSGA2TestWorkflow
    cls = type("EvoXTestWorkflow", (base,), {})
    env = ToyEnv(2) if objectives is None else VectorRewardEnv(2, objectives)
    with patch(
        "evorl.algorithms.contrib.evox.evox_workflow.create_env", return_value=env
    ):
        config.torch_compile = jit
        workflow = cls.build_from_config(config, jit=jax_jit)
    workflow.recorder = Mock()
    workflow.checkpoint_manager = Mock()
    try:
        assert isinstance(workflow, Workflow)
        assert workflow.num_devices == 1
        assert hasattr(workflow._evaluate, "lower") is jax_jit
        assert hasattr(workflow.algorithm.evaluate_fn, "lower") is jax_jit
        assert not hasattr(base._evaluate, "lower")
        assert workflow.algorithm._compile is jit
        assert workflow.mesh is None
        assert workflow.config is not config
        assert OmegaConf.is_readonly(workflow.config)
        assert not OmegaConf.is_readonly(config)
        state = workflow.init(jax.random.key(5))
        snapshot = jax.tree.map(
            lambda x: np.array(x, copy=True), workflow_state_with_key_data(state)
        )
        # Initialization is complete before the first ordinary update.
        workflow.recorder.write.assert_not_called()
        workflow.checkpoint_manager.save.assert_not_called()
        assert int(state.metrics.iterations) == 0
        assert int(state.metrics.sampled_episodes) == 16
        second_metrics, second = workflow.step(state)
        chex.assert_trees_all_equal(workflow_state_with_key_data(state), snapshot)
        workflow.recorder.write.assert_not_called()
        workflow.checkpoint_manager.save.assert_not_called()
        # CSO evaluates only its two students after initialization.
        candidates = 2 if objectives is None else 4
        assert second_metrics.objectives.shape == (
            (candidates,) if objectives is None else (candidates, objectives)
        )
        assert int(second.metrics.iterations) == 1
        assert int(second.metrics.sampled_episodes) == 16 + 4 * candidates
        save(tmp_path / "workflow", state)
        restored = load(tmp_path / "workflow", state)
        resumed = workflow.learn(restored)
        chex.assert_trees_all_equal(
            workflow_state_with_key_data(resumed), workflow_state_with_key_data(second)
        )
        workflow.checkpoint_manager.save.assert_called_once()
        workflow.checkpoint_manager.save.assert_called_with(1, resumed, force=True)
        assert workflow.learn(resumed) is resumed
        logged = workflow.recorder.write.call_args.args[0]["objectives"]
        if objectives is not None:
            assert set(logged) == {f"objective_{i}" for i in range(objectives)}
    finally:
        workflow.close()


@pytest.mark.integration
@pytest.mark.parametrize("num_iters", [0, 1])
def test_workflow_initialization_respects_iteration_budget(num_iters):
    config = workflow_config()
    config.num_iters = num_iters
    cls = type("BudgetEvoXWorkflow", (get_class(config.workflow_cls),), {})
    with patch(
        "evorl.algorithms.contrib.evox.evox_workflow.create_env", return_value=ToyEnv(2)
    ):
        workflow = cls.build_from_config(config)
    workflow.recorder = Mock()
    workflow.checkpoint_manager = Mock()
    try:
        with patch.object(
            workflow.algorithm, "evaluate_fn", wraps=workflow.algorithm.evaluate_fn
        ) as evaluate:
            state = workflow.init(jax.random.key(8))
            assert int(state.metrics.iterations) == 0
            assert int(state.metrics.sampled_episodes) == 16
            assert evaluate.call_count == 1
            assert state.algorithm_state is not None
            workflow.recorder.write.assert_not_called()
            workflow.checkpoint_manager.save.assert_not_called()
            result = workflow.learn(state)
            assert int(result.metrics.iterations) == num_iters
            assert int(result.metrics.sampled_episodes) == 16 + 8 * num_iters
            assert evaluate.call_count == 1 + num_iters
            assert workflow.learn(result) is result
            assert evaluate.call_count == 1 + num_iters
            if num_iters:
                workflow.checkpoint_manager.save.assert_called_once_with(
                    1, result, force=True
                )
                assert workflow.recorder.write.call_count == 2
            else:
                assert result is state
                workflow.checkpoint_manager.save.assert_not_called()
                workflow.recorder.write.assert_not_called()
    finally:
        workflow.close()


def test_workflow_rejects_negative_budget_before_creating_environment():
    config = workflow_config()
    config.num_iters = -1
    cls = type("InvalidBudgetEvoXWorkflow", (get_class(config.workflow_cls),), {})
    with (
        patch("evorl.algorithms.contrib.evox.evox_workflow.create_env") as make_env,
        pytest.raises(ValueError, match="num_iters must be nonnegative"),
    ):
        cls.build_from_config(config)
    make_env.assert_not_called()


@pytest.mark.integration
def test_workflow_observation_warmup():
    config = workflow_config()
    config.normalize_obs = True
    config.random_timesteps = 4
    base = get_class(config.workflow_cls)
    cls = type("EvoXWarmupWorkflow", (base,), {})
    with (
        patch(
            "evorl.algorithms.contrib.evox.evox_workflow.create_env",
            return_value=ToyEnv(2),
        ),
        patch("evorl.algorithms.ec.obs_utils.create_env", return_value=ToyEnv(2)),
        patch(
            "evorl.algorithms.contrib.evox.evox_workflow.init_obs_preprocessor",
            wraps=init_obs_preprocessor,
        ) as initialize,
    ):
        workflow = cls.build_from_config(config)
        try:
            state = workflow.setup(jax.random.key(3))
            initialize.assert_called_once()
            assert not any(
                isinstance(value, jax.core.Tracer)
                for value in jax.tree.leaves(initialize.call_args.args)
            )
            assert (
                state.algorithm_state.evaluation_state.agent_state.obs_preprocessor_state.count
                > 0
            )
            metrics, _ = workflow.step(state)
            assert np.all(np.isfinite(metrics.objectives))
        finally:
            workflow.close()


def test_workflow_rejects_multi_device_and_mismatched_rewards():
    config = workflow_config()
    base = get_class(config.workflow_cls)
    cls = type("InvalidEvoXWorkflow", (base,), {})
    with (
        patch("evorl.algorithms.contrib.evox.evox_workflow.create_env") as make_env,
        pytest.raises(NotImplementedError, match="one device"),
    ):
        cls.build_from_config(config, multi_device=True)
    make_env.assert_not_called()
    with (
        patch(
            "evorl.algorithms.contrib.evox.evox_workflow.create_env",
            return_value=VectorRewardEnv(2),
        ),
        pytest.raises(ValueError, match="CSO requires scalar rewards"),
    ):
        cls.build_from_config(config)


@pytest.mark.integration
def test_workflow_checkpoint_schedule_and_resume(tmp_path):
    config = workflow_config()
    config.num_iters = 4
    config.output_dir = str(tmp_path)
    config.checkpoint.enable = True
    config.checkpoint.save_interval_steps = 3
    cls = type("CheckpointEvoXWorkflow", (get_class(config.workflow_cls),), {})
    with patch(
        "evorl.algorithms.contrib.evox.evox_workflow.create_env",
        return_value=ToyEnv(2),
    ):
        workflow = cls.build_from_config(config)
    workflow.recorder = Mock()
    try:
        state = workflow.learn(workflow.setup(jax.random.key(11)))
        workflow.checkpoint_manager.wait_until_finished()
        # Orbax saves the first step, interval steps and our forced final step.
        assert workflow.checkpoint_manager.all_steps() == [1, 3, 4]
        restored = workflow.checkpoint_manager.restore(4, state)
        _, expected = workflow.step(state)
        _, actual = workflow.step(restored)
        chex.assert_trees_all_equal(
            workflow_state_with_key_data(expected), workflow_state_with_key_data(actual)
        )
    finally:
        workflow.close()


@pytest.mark.integration
@pytest.mark.parametrize("objectives", [2, 3])
def test_rvea_workflow_keeps_int32_generation_state(objectives, jit):
    config = workflow_config("rvea")
    config.torch_compile = jit
    config.fr = 0.5
    cls = type("RVEATestWorkflow", (get_class(config.workflow_cls),), {})
    with patch(
        "evorl.algorithms.contrib.evox.evox_workflow.create_env",
        return_value=VectorRewardEnv(2, objectives),
    ):
        workflow = cls.build_from_config(config)
    try:
        state = workflow.init(jax.random.key(2))
        for generation in range(4):
            tensors = state.algorithm_state.params_and_buffers
            assert tensors["gen"].dtype == jnp.int32
            assert tensors["max_gen"].dtype == jnp.int32
            assert int(tensors["gen"]) == generation
            if generation < 3:
                _, state = workflow.step(state)
    finally:
        workflow.close()


def test_int32_rvea_matches_native_algorithm_updates():
    device = torch_device()
    kwargs = dict(
        pop_size=8,
        n_objs=2,
        lb=-torch.ones(3, device=device),
        ub=torch.ones(3, device=device),
        device=device,
        max_gen=4,
        fr=0.5,
    )

    def evaluate(pop):
        return torch.stack((pop.square().sum(-1), (pop - 0.5).square().sum(-1)), -1)

    algorithms = []
    for algorithm_cls in (EvoXRVEA, RVEA):
        with fork_torch_rng(device, seed=7), torch.device(device):
            algorithm = algorithm_cls(**kwargs)
            algorithm.evaluate = evaluate
            algorithm.init_step()
            algorithms.append(algorithm)
    for _ in range(4):
        for algorithm in algorithms:
            with fork_torch_rng(device, seed=11), torch.device(device):
                algorithm.step()
        original, adapted = algorithms
        assert adapted.gen.dtype == torch.int32
        assert adapted.max_gen.dtype == torch.int32
        for name in ("pop", "fit", "reference_vector", "rv_adapt_every"):
            torch.testing.assert_close(
                getattr(original, name), getattr(adapted, name), equal_nan=True
            )


@pytest.mark.parametrize("objectives", [None, 1])
def test_rvea_rejects_scalar_or_size_one_vector(objectives):
    config = workflow_config("rvea")
    base = get_class(config.workflow_cls)
    cls = type("InvalidRVEAWorkflow", (base,), {})
    env = ToyEnv(2) if objectives is None else VectorRewardEnv(2, objectives)
    with (
        patch(
            "evorl.algorithms.contrib.evox.evox_workflow.create_env", return_value=env
        ),
        pytest.raises(ValueError, match="at least two objectives"),
    ):
        cls.build_from_config(config)


@pytest.mark.integration
def test_workflow_counts_multiple_batches_and_advances_key_for_each():
    config = workflow_config()
    cls = type("MultipleEvaluationTestWorkflow", (MultipleEvaluationWorkflow,), {})
    with patch(
        "evorl.algorithms.contrib.evox.evox_workflow.create_env", return_value=ToyEnv(2)
    ):
        workflow = cls.build_from_config(config)
    try:
        state = workflow.setup(jax.random.key(4))
        assert int(state.metrics.iterations) == 0
        assert int(state.metrics.sampled_episodes) == 6 * 4
        initial_key = jax.random.split(jax.random.key(4), 4)[0]
        expected_initial_key = jax.random.split(jax.random.split(initial_key)[0])[0]
        chex.assert_trees_all_equal(
            jax.random.key_data(state.algorithm_state.evaluation_state.key),
            jax.random.key_data(expected_initial_key),
        )
        metrics, result = workflow.step(state)
        assert metrics.objectives.shape == (6,)
        assert int(result.metrics.iterations) == 1
        assert int(result.metrics.sampled_episodes) == 2 * 6 * 4
        expected_key = jax.random.split(
            jax.random.split(state.algorithm_state.evaluation_state.key)[0]
        )[0]
        chex.assert_trees_all_equal(
            jax.random.key_data(result.algorithm_state.evaluation_state.key),
            jax.random.key_data(expected_key),
        )
    finally:
        workflow.close()


@pytest.mark.integration
def test_workflow_evaluates_host_environments(host_envs):
    backend, _ = host_envs
    config = workflow_config()
    config.env = dict(env_name="Pendulum-v1", env_type=backend, max_episode_steps=2)
    base = get_class(config.workflow_cls)
    cls = type("HostEvoXWorkflow", (base,), {})
    workflow = cls.build_from_config(config)
    try:
        state = workflow.setup(jax.random.key(1))
        metrics, state = workflow.step(state)
        assert np.all(np.isfinite(metrics.objectives))
        assert int(state.metrics.sampled_episodes) == 24
    finally:
        workflow.close()


@pytest.mark.integration
@pytest.mark.parametrize("torch_compile", [False, True])
def test_evox_trains_real_brax_scalar_and_vector_rewards(torch_compile):
    root = Path(__file__).resolve().parents[1]
    for vector in (False, True):
        config = workflow_config()
        config.torch_compile = torch_compile
        config.env = OmegaConf.load(
            root / f"configs/env/brax/{'ant_multi_objective' if vector else 'ant'}.yaml"
        )
        config.env.max_episode_steps = 2
        config.num_envs = 1
        config.episodes_for_fitness = 1
        base = NSGA2TestWorkflow if vector else get_class(config.workflow_cls)
        cls = type("BraxEvoXWorkflow", (base,), {})
        workflow = cls.build_from_config(config)
        try:
            state = workflow.setup(jax.random.key(2))
            metrics, state = workflow.step(state)
            assert np.all(np.isfinite(metrics.objectives))
            assert int(state.metrics.iterations) == 1
        finally:
            workflow.close()


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["cso", "rvea"])
@pytest.mark.parametrize("torch_compile", [False, True])
def test_evox_brax_full_training_and_checkpoint_resume(
    algorithm, torch_compile, tmp_path
):
    root = Path(__file__).resolve().parents[1]
    env_name = "ant" if algorithm == "cso" else "ant_multi_objective"
    config = OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / f"configs/agent/contrib/{algorithm}-evox.yaml"),
        {
            "env": OmegaConf.load(root / f"configs/env/brax/{env_name}.yaml"),
            "num_iters": 5,
            "pop_size": 16,
            "num_envs": 4,
            # Exercise rounding to a full batch of four episodes.
            "episodes_for_fitness": 3,
            "torch_compile": torch_compile,
            "output_dir": str(tmp_path),
            "checkpoint": {"enable": True, "save_interval_steps": 2},
        },
    )
    if algorithm == "rvea":
        # Cover both branches of native reference-vector adaptation.
        config.fr = 0.5
    cls = type("FullBraxEvoXWorkflow", (get_class(config.workflow_cls),), {})
    before = rng_states()
    workflow = cls.build_from_config(config)
    log_path = tmp_path / "training.log"
    workflow.add_recorders([LogRecorder(str(log_path), console=False)])
    try:
        state = workflow.init(jax.random.key(42))
        assert int(state.metrics.iterations) == 0
        assert int(state.metrics.sampled_episodes) == 16 * 4
        assert log_path.read_text() == ""
        assert workflow.checkpoint_manager.all_steps() == []
        observation_statistics = (
            state.algorithm_state.evaluation_state.agent_state.obs_preprocessor_state
        )
        assert int(observation_statistics.count) == config.random_timesteps
        final = workflow.learn(state)
        assert int(final.metrics.iterations) == 5
        candidates_per_update = 8 if algorithm == "cso" else 16
        assert int(final.metrics.sampled_episodes) == 4 * (
            16 + 5 * candidates_per_update
        )
        assert float(final.metrics.sampled_timesteps_m) > 0
        tensors = final.algorithm_state.params_and_buffers
        assert all(value.device == jax.local_devices()[0] for value in tensors.values())
        population, fitness = np.asarray(tensors["pop"]), np.asarray(tensors["fit"])
        if algorithm == "cso":
            assert np.all(np.isfinite(population))
            assert np.all(np.isfinite(fitness))
        else:
            # RVEA reserves NaN rows for reference vectors without survivors.
            valid = ~np.isnan(population).any(axis=-1)
            assert valid.any()
            assert np.all(np.isfinite(population[valid]))
            assert np.all(np.isfinite(fitness[valid]))
            assert np.array_equal(~valid, np.isnan(fitness).all(axis=-1))
            assert int(tensors["gen"]) == 5
            assert tensors["gen"].dtype == jnp.int32
            assert tensors["max_gen"].dtype == jnp.int32
        text = log_path.read_text()
        assert text.count("iteration ") == 10
        for iteration in range(1, 6):
            assert text.count(f"iteration {iteration}:") == 2
        if algorithm == "cso":
            assert "objective_" not in text
        else:
            assert "reward_forward" in text
            assert "reward_ctrl" in text
        workflow.checkpoint_manager.wait_until_finished()
        assert workflow.checkpoint_manager.all_steps() == [1, 2, 4, 5]
        restored = workflow.checkpoint_manager.restore(2, final)
        workflow.checkpoint_manager.close()
        workflow.checkpoint_manager = DummyCheckpointManager()
        resumed = workflow.learn(restored)
        chex.assert_trees_all_equal(
            workflow_state_with_key_data(resumed), workflow_state_with_key_data(final)
        )
        assert workflow.learn(resumed) is resumed
        assert_rng_states_equal(before)
    finally:
        workflow.close()
