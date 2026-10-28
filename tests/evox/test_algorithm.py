"""Functional EvoX state, native update equivalence and failure isolation."""

from unittest.mock import Mock, patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch", minversion="2.9.0")
pytest.importorskip("evox")

import weakref
from evox.algorithms import CSO, NSGA2, RVEA as EvoXRVEA
from evox.core import Algorithm, Mutable, Parameter
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    DataDependentOutputException,
)

from evorl.algorithms.contrib.evox.rvea import RVEA
from evorl.utils.orbax_utils import CheckpointManager
from evorl.utils.torch_utils import fork_torch_rng
from evorl.types import PyTreeDict
from tests.helpers.evox import (
    make_wrapper,
    torch_device,
    rng_states,
    assert_rng_states_equal,
    evaluate_rewards,
)


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
        fitnesses = population.sum(-1)
        return (
            {
                "scalar": fitnesses.sum(),
                "candidates": fitnesses[:-1],
                "rank": fitnesses[:, None, None],
                "dtype": fitnesses.astype(jnp.int32),
                "empty": fitnesses[:, None][:, :0],
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
        fitnesses = -jnp.square(population).sum(-1)
        return (
            (
                jnp.stack((fitnesses, fitnesses - 1), -1)
                if multi_objective
                else fitnesses
            ),
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
