"""Utilities for optional PyTorch integration."""

from contextlib import contextmanager
from collections.abc import Iterator

import jax
import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensor


def torch_to_jax(tensor: torch.Tensor, *, copy: bool = True) -> jax.Array:
    """Convert a real PyTorch tensor to JAX on the same device using DLPack.

    The default copy detaches autograd and creates independent, contiguous
    storage before export. Later PyTorch mutations cannot affect the JAX array.
    This function runs outside JAX transformations.

    Args:
        tensor: Real, dense PyTorch tensor. Fake and meta tensors have no
            storage to export. Its dtype must be representable with JAX's
            configured precision; conversion never silently narrows 64-bit data.
        copy: Copy the input by default. With `False`, share storage when the
            layout permits; callers must keep that storage immutable while JAX
            uses it. Incompatible layouts raise instead of silently copying.

    Returns:
        A JAX array with the same shape, dtype and device as the input.

    Raises:
        TypeError: If the input is fake/meta or requires disabled JAX precision.
    """
    if isinstance(tensor, FakeTensor):
        raise TypeError("Cannot convert a FakeTensor to JAX; real tensors are required")
    if tensor.is_meta:
        raise TypeError(
            "Cannot convert a meta tensor to JAX; real tensors are required"
        )
    if not jax.config.x64_enabled and tensor.dtype in (
        torch.int64,
        torch.uint64,
        torch.float64,
        torch.complex128,
    ):
        raise TypeError(
            f"Cannot convert 64-bit dtype {tensor.dtype} with JAX x64 disabled"
        )
    tensor = tensor.detach()
    if copy:
        tensor = tensor.clone(memory_format=torch.contiguous_format)
    return jax.dlpack.from_dlpack(tensor, copy=False)


def jax_to_torch(array: jax.Array, *, copy: bool = True) -> torch.Tensor:
    """Convert a concrete JAX array to PyTorch on the same device using DLPack.

    This function runs outside JAX transformations and does not provide
    autograd across frameworks.

    Args:
        array: Concrete JAX array on a device supported by PyTorch.
        copy: Copy by default so PyTorch can safely mutate the result. With
            `False`, share storage and keep the result read-only; a subsequent
            out-of-place operation can create independent, writable storage.

    Returns:
        A PyTorch tensor with the same shape, dtype and device as the input.
    """
    tensor = torch.from_dlpack(array)
    return tensor.clone() if copy else tensor


def jax_to_torch_device(device: jax.Device) -> torch.device:
    """Find the PyTorch device corresponding to a local JAX CPU or GPU.

    GPU devices use the available PyTorch accelerator type and JAX's local
    hardware index, rather than PyTorch's current device index. Both frameworks
    must use the same visible devices in the same order.

    Args:
        device: Local device holding the JAX arrays shared with PyTorch.

    Returns:
        The corresponding CPU or accelerator device in PyTorch.

    Raises:
        ValueError: If the JAX platform is unsupported or no PyTorch
            accelerator is available for a JAX GPU.
    """
    if device.platform == "cpu":
        return torch.device("cpu")
    if device.platform != "gpu":
        raise ValueError(f"Unsupported JAX platform for PyTorch: {device.platform}")
    accelerator = torch.accelerator.current_accelerator(check_available=True)
    if accelerator is None:
        raise ValueError("No PyTorch accelerator is available for the JAX GPU")
    return torch.device(accelerator.type, device.local_hardware_id)


def get_torch_rng_state(
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Copy the default CPU and selected accelerator RNG states to NumPy.

    Args:
        device: CPU or an accelerator device with an explicit local index.

    Returns:
        CPU RNG bytes and accelerator RNG bytes. The second entry is `None`
        for CPU execution. These independent arrays can be checkpointed.
    """
    cpu_state = torch.random.get_rng_state().numpy().copy()
    device_state = (
        None
        if device.type == "cpu"
        else torch.get_device_module(device).get_rng_state(device.index).numpy().copy()
    )
    return cpu_state, device_state


@contextmanager
def fork_torch_rng(
    device: torch.device,
    *,
    seed: int | None = None,
    state: tuple[np.ndarray, np.ndarray | None] | None = None,
) -> Iterator[None]:
    """Temporarily seed or restore the default CPU and selected device RNGs.

    Both generators return to their previous states on exit, including when
    the body raises. Other accelerators are untouched. Calls must be sequential
    because these generators belong to the process, rather than the context.

    Args:
        device: CPU or an accelerator device with an explicit local index.
        seed: Seed both generators before entering the body.
        state: CPU/device RNG bytes from `get_torch_rng_state`. Use either
            `seed` or `state`. When neither is provided, only isolate RNG use.

    Yields:
        Control with the temporary default RNG states active.

    Raises:
        ValueError: If both `seed` and `state` are supplied, or accelerator
            execution receives no accelerator RNG state.
    """
    if seed is not None and state is not None:
        raise ValueError("Specify either seed or state, not both")
    devices = [] if device.type == "cpu" else [device.index]
    with torch.random.fork_rng(devices=devices, device_type=device.type):
        if seed is not None:
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.accelerator.device_index(device.index):
                    torch.get_device_module(device).manual_seed(seed)
        elif state is not None:
            cpu_state, device_state = state
            torch.random.set_rng_state(torch.from_numpy(cpu_state.copy()))
            if devices:
                if device_state is None:
                    raise ValueError("An accelerator RNG state is required")
                torch.get_device_module(device).set_rng_state(
                    torch.from_numpy(device_state.copy()), device.index
                )
        yield
