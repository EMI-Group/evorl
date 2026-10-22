"""Run PyTorch EvoX algorithms with explicit, checkpointable state."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._pytree import tree_leaves
from evox.core import (
    Algorithm,
    use_state,
    compile as torch_compile,
)

from evorl.types import PyTreeData
from evorl.utils.torch_utils import (
    fork_torch_rng,
    get_torch_rng_state,
    jax_to_torch,
    torch_to_jax,
)


def _check_tensor_dtype(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.dtype in (torch.int64, torch.uint64, torch.float64, torch.complex128):
        raise TypeError(
            f"{name} has unsupported 64-bit dtype {tensor.dtype}; "
            "adjust the EvoX algorithm to use a supported dtype"
        )


class EvoXAlgorithmState(PyTreeData):
    """Complete persistent state for the algorithm and its JAX evaluation.

    Algorithm tensors remain on their execution device. RNG byte arrays stay
    on the host. Python consumes this state outside JAX transforms, and it can
    be checkpointed directly.
    Algorithms with 64-bit tensors are rejected without casting their state.
    """

    params_and_buffers: dict[str, jax.Array]
    cpu_rng_state: np.ndarray
    device_rng_state: np.ndarray | None
    evaluation_state: Any


class EvoXAlgorithmWrapper:
    """Run EvoX algorithms with explicit, checkpointable algorithm state.

    The algorithm class and explicit constructor arguments define the static
    configuration. Construction prepares a FakeTensor algorithm template with
    real, fixed tensor attributes. Each `init(key, evaluation_state)` constructs
    a fresh algorithm and runs its native `init_step`, returning state ready for
    regular `step` calls without changing the template or execution functions.
    Evolving algorithm tensors must be registered parameters or buffers,
    as required by EvoX's `Parameter`/`Mutable` convention. Non-persistent buffers
    are included. Unregistered Python attributes and custom RNG generators are
    not a supported location for evolving state. 64-bit tensors are rejected;
    their dtypes must be adjusted in the algorithm, without wrapper conversion.

    `step` runs on the Python host. The supplied JAX evaluator is responsible
    for its own JIT compilation. `compile=True` separately compiles PyTorch
    updates, with a graph break around evaluation. DLPack keeps populations and
    fitness on the same device; copies protect immutable JAX input arrays from
    PyTorch in-place updates. Input state and process RNG are restored on both
    success and failure. Calls must be sequential, outside JAX transforms.
    """

    def __init__(
        self,
        algorithm_cls: type[Algorithm],
        algorithm_kwargs: Mapping[str, Any],
        device: torch.device,
        compile: bool = False,
        *,
        evaluate: Callable[[jax.Array, Any], tuple[jax.Array, Any, Any]],
        static_tensors: Mapping[str, torch.Tensor] | None = None,
    ):
        """Prepare the fixed algorithm template and functional update functions.

        Registered parameters and buffers remain FakeTensor placeholders until
        `use_state` temporarily replaces them during execution. The constructor
        must support FakeTensor operations without inspecting tensor values.

        Args:
            algorithm_cls: Concrete EvoX algorithm class. Random initialization
                must use PyTorch's default generators.
            algorithm_kwargs: Complete keyword arguments for `algorithm_cls`,
                including its device argument when needed. Arguments are copied
                when stored and for each construction, so caller changes and
                constructor mutations cannot affect later initializations.
                Values must support `copy.deepcopy`. No constructor arguments
                are inferred, injected or overridden by this wrapper. Algorithms
                are constructed under `torch.device(device)` so new tensors
                without an explicit device use it too. Existing tensor arguments
                must already be on the intended device.
            device: CPU or current accelerator device shared with the evaluator.
            compile: Compile PyTorch updates with EvoX's `torch.compile`
                wrapper. JAX evaluation remains outside the PyTorch graph.
            evaluate: Fixed JAX reward function `(population, evaluation_state)`
                returning `(rewards, metrics, next_evaluation_state)`. Rewards are
                maximized and have shape `(candidates,)` or
                `(candidates, objectives)`. Dynamic evaluation data is supplied
                to `init` and retained in the returned state. Metrics are
                per-call outputs, not persistent state. This callable must not
                mutate its inputs.
            static_tensors: Real, immutable tensor attributes for the template,
                keyed by attribute paths such as `lb` or `operator.weights`.
                Shapes and dtypes must match the constructor's attributes.
                Registered parameters/buffers cannot be overridden here.
                Values are copied onto `device`. Every unregistered FakeTensor
                attribute must be replaced; none may reach actual evaluation.
        """
        if not isinstance(algorithm_cls, type) or not issubclass(
            algorithm_cls, Algorithm
        ):
            raise TypeError("algorithm_cls must be an evox.core.Algorithm subclass")
        self.algorithm_cls = algorithm_cls
        self._algorithm_kwargs = deepcopy(dict(algorithm_kwargs))
        self.device = torch.device(device)
        if self.device.type != "cpu":
            accelerator = torch.accelerator.current_accelerator(check_available=True)
            if accelerator is None or accelerator.type != self.device.type:
                raise ValueError(
                    "device must use CPU or the current PyTorch accelerator"
                )
            if self.device.index is None:
                self.device = torch.device(
                    accelerator.type, torch.accelerator.current_device_index()
                )
        self._compile = compile
        self.evaluate_fn = evaluate
        self._evaluation_state = None
        self._evaluation_metrics = None
        # Copy real tensor arguments before FakeTensorMode intercepts storage
        # operations; copying inside the mode fails on PyTorch 2.9.
        constructor_kwargs = deepcopy(self._algorithm_kwargs)
        with (
            fork_torch_rng(self.device, seed=0),
            torch.no_grad(),
            torch.device(self.device),
            FakeTensorMode(
                allow_non_fake_inputs=True, allow_fallback_kernels=False
            ) as fake_mode,
        ):
            self._algorithm = algorithm_cls(**constructor_kwargs)
            # A constructor may register an input tensor without computing on
            # it. Preserve shared tensors when making FakeTensor placeholders.
            placeholders = {}
            for module in self._algorithm.modules():
                for fields in (module._parameters, module._buffers):
                    for name, value in fields.items():
                        if value is None:
                            continue
                        identity = id(value)
                        if identity not in placeholders:
                            placeholder = fake_mode.from_tensor(value)
                            if isinstance(value, torch.nn.Parameter) and not isinstance(
                                placeholder, torch.nn.Parameter
                            ):
                                placeholder = torch.nn.Parameter(
                                    placeholder, requires_grad=value.requires_grad
                                )
                            placeholders[identity] = placeholder
                        fields[name] = placeholders[identity]
        for name, value in (
            dict(self._algorithm.named_parameters())
            | dict(self._algorithm.named_buffers())
        ).items():
            _check_tensor_dtype(value, name=f"Algorithm tensor {name!r}")
            if value.device != self.device:
                raise ValueError(
                    f"Algorithm tensor {name!r} is on {value.device}; "
                    f"construct it on {self.device}"
                )
        for path, value in deepcopy(dict(static_tensors or {})).items():
            module_path, _, name = path.rpartition(".")
            module = self._algorithm.get_submodule(module_path)
            if name in module._parameters or name in module._buffers:
                raise ValueError(f"{path!r} is registered state, not a static tensor")
            original = getattr(module, name)
            if not isinstance(original, torch.Tensor):
                raise TypeError(f"Static attribute {path!r} must already be a tensor")
            if not isinstance(value, torch.Tensor) or isinstance(value, FakeTensor):
                raise TypeError(f"Static tensor {path!r} must be a real PyTorch tensor")
            if value.shape != original.shape or value.dtype != original.dtype:
                raise ValueError(
                    f"Static tensor {path!r} has a mismatched shape or dtype"
                )
            _check_tensor_dtype(value, name=f"Static tensor {path!r}")
            setattr(module, name, value.detach().to(self.device))
        for prefix, module in self._algorithm.named_modules():
            for name, value in vars(module).items():
                if name in ("_parameters", "_buffers", "_modules"):
                    continue
                if any(isinstance(leaf, FakeTensor) for leaf in tree_leaves(value)):
                    path = f"{prefix}.{name}" if prefix else name
                    raise ValueError(
                        f"Unregistered FakeTensor attribute {path!r}; "
                        "provide its real value through static_tensors"
                    )
        self._state_shapes = {
            name: value.shape
            for name, value in (
                dict(self._algorithm.named_parameters())
                | dict(self._algorithm.named_buffers())
            ).items()
        }
        self._algorithm.evaluate = self._evaluate
        self._init_step = use_state(self._algorithm.init_step, strict=True)
        self._step = use_state(self._algorithm.step, strict=True)
        if compile:
            self._init_step = torch_compile(self._init_step, fullgraph=False)
            self._step = torch_compile(self._step, fullgraph=False)

    def init(
        self, key: jax.Array, evaluation_state: Any = None
    ) -> tuple[tuple[Any, ...], EvoXAlgorithmState]:
        """Return state after EvoX's native first call and its evaluations.

        `init_step` may evaluate the initial population or perform a complete
        optimization update, according to the supplied EvoX algorithm. Its
        evaluations advance the caller's explicit evaluation state. Only the
        default RNG states are temporarily changed; execution functions stay
        fixed after construction.

        Args:
            key: Seeds the algorithm constructor and native first call.
            evaluation_state: Dynamic input to the fixed reward function.

        Returns:
            A tuple of metrics from every evaluation batch, followed by the
            complete initialized state. Subsequent calls use `step(state)`.
        """
        seed = int(jax.random.bits(key, dtype="uint32"))
        with (
            fork_torch_rng(self.device, seed=seed),
            torch.no_grad(),
            torch.device(self.device),
        ):
            algorithm = self.algorithm_cls(**deepcopy(self._algorithm_kwargs))
            tensors = dict(algorithm.named_parameters()) | dict(
                algorithm.named_buffers()
            )
            return self._run_step(self._init_step, tensors, evaluation_state)

    def step(
        self, state: EvoXAlgorithmState
    ) -> tuple[tuple[Any, ...], EvoXAlgorithmState]:
        """Run a regular EvoX update using the fixed reward function.

        Args:
            state: Complete state from `init` or a checkpoint, including the
                evaluation state. Arrays are never donated or modified. EvoX
                may evaluate several batches of different candidate counts.

        Returns:
            A tuple of per-batch evaluation metrics and the complete next state.
            All reward objectives are maximized, including size-one vectors.
        """
        tensors = {
            name: jax_to_torch(value).to(self.device)
            for name, value in state.params_and_buffers.items()
        }
        with (
            fork_torch_rng(
                self.device, state=(state.cpu_rng_state, state.device_rng_state)
            ),
            torch.no_grad(),
        ):
            return self._run_step(self._step, tensors, state.evaluation_state)

    def _run_step(self, step_fn, tensors, evaluation_state):
        """Execute either native method and export complete state and batch metrics.

        The caller supplies fresh working tensors and the seeded/restored RNG
        scope. Initialization and ordinary updates share this callback lifecycle
        and state export without converting the initial tensors through JAX.
        """
        if tensors.keys() != self._state_shapes.keys() or any(
            value.shape != self._state_shapes[name] for name, value in tensors.items()
        ):
            raise ValueError("Algorithm state does not match this wrapper")
        for name, value in tensors.items():
            _check_tensor_dtype(value, name=f"Algorithm tensor {name!r}")
            if value.device != self.device:
                raise ValueError(
                    f"Algorithm tensor {name!r} is on {value.device}; "
                    f"construct it on {self.device}"
                )
        # EvoX calls evaluate(population) without a context argument. Thread the
        # caller's explicit state through that callback for this synchronous call.
        self._evaluation_state = evaluation_state
        self._evaluation_metrics = []
        try:
            tensors = step_fn(tensors)
            for name, value in tensors.items():
                _check_tensor_dtype(value, name=f"Algorithm tensor {name!r}")
            cpu_rng_state, device_rng_state = get_torch_rng_state(self.device)
            state = EvoXAlgorithmState(
                params_and_buffers={
                    name: torch_to_jax(value) for name, value in tensors.items()
                },
                cpu_rng_state=cpu_rng_state,
                device_rng_state=device_rng_state,
                evaluation_state=self._evaluation_state,
            )
            return tuple(self._evaluation_metrics), state
        finally:
            self._evaluation_state = None
            self._evaluation_metrics = None

    @torch.compiler.disable
    def _evaluate(self, population: torch.Tensor) -> torch.Tensor:
        if not isinstance(population, torch.Tensor) or population.ndim != 2:
            raise ValueError("Expected a dense population with shape (candidates, dim)")
        _check_tensor_dtype(population, name="Population")
        fitnesses, metrics, self._evaluation_state = self.evaluate_fn(
            torch_to_jax(population), self._evaluation_state
        )
        if (
            not isinstance(fitnesses, jax.Array)
            or fitnesses.ndim not in (1, 2)
            or fitnesses.shape[0] != population.shape[0]
            or not jnp.issubdtype(fitnesses.dtype, jnp.floating)
            or (fitnesses.ndim == 2 and fitnesses.shape[1] == 0)
        ):
            raise ValueError(
                "Fitness must be a floating JAX array with shape (candidates,) "
                "or (candidates, objectives)"
            )
        # Negation below creates independent storage, so avoid an earlier copy.
        result = jax_to_torch(fitnesses, copy=False)
        _check_tensor_dtype(result, name="JAX fitness")
        if result.device != population.device:
            raise ValueError("JAX fitness and the EvoX population must share a device")
        if self._evaluation_metrics is not None:
            self._evaluation_metrics.append(metrics)
        # All EvoRL objectives are maximized. EvoX algorithms minimize.
        # Negation also creates independent storage for algorithm mutations.
        return -result
