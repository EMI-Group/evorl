"""Device placement, tensor conversion and RNG isolation for PyTorch."""

from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch", minversion="2.9.0")
from torch._subclasses.fake_tensor import FakeTensorMode

from evorl.utils.torch_utils import (
    fork_torch_rng,
    get_torch_rng_state,
    jax_to_torch,
    jax_to_torch_device,
    torch_to_jax,
)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int32, torch.bool]
)
def test_tensor_conversion_preserves_shape_dtype_device_and_values(dtype):
    device = jax.local_devices()[0]
    tensor = (torch.arange(6, device=jax_to_torch_device(device)) % 2).to(dtype)
    tensor = tensor.reshape(2, 3)
    array = torch_to_jax(tensor)
    restored = jax_to_torch(array)
    assert array.shape == tensor.shape
    assert array.device == device
    assert restored.device == tensor.device
    assert restored.dtype == tensor.dtype
    torch.testing.assert_close(restored, tensor)


def test_torch_to_jax_copies_noncontiguous_autograd_tensor():
    device = jax_to_torch_device(jax.local_devices()[0])
    tensor = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4).T
    tensor.requires_grad_(True)
    assert not tensor.is_contiguous()
    expected = tensor.detach().cpu().numpy().copy()
    array = torch_to_jax(tensor)
    array.block_until_ready()
    with torch.no_grad():
        tensor.add_(100)
    np.testing.assert_array_equal(array, expected)
    # The exported array is an ordinary input to compiled JAX computations.
    np.testing.assert_allclose(
        jax.jit(lambda x: (x * 2).sum())(array), expected.sum() * 2
    )


def test_jax_to_torch_copy_protects_jax_input_from_mutation():
    array = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    expected = np.array(array, copy=True)
    tensor = jax_to_torch(array)
    assert not tensor.requires_grad
    tensor.fill_(100)
    # The later JAX read must still see the original values.
    np.testing.assert_array_equal(jax.jit(lambda x: x + 1)(array), expected + 1)


def test_explicit_no_copy_shares_storage_in_both_directions():
    tensor = torch.arange(
        6, dtype=torch.float32, device=jax_to_torch_device(jax.local_devices()[0])
    )
    array = torch_to_jax(tensor, copy=False)
    alias = jax_to_torch(array, copy=False)
    assert alias.data_ptr() == tensor.data_ptr()
    torch.testing.assert_close(alias, tensor)

    array = jnp.arange(6, dtype=jnp.float32)
    first = jax_to_torch(array, copy=False)
    second = torch.from_dlpack(array)
    assert first.data_ptr() == second.data_ptr()
    # Shared tensors are deliberately not mutated.


@pytest.mark.parametrize("kind", ["fake", "meta"])
def test_torch_to_jax_rejects_tensors_without_real_storage_before_dlpack(kind):
    if kind == "fake":
        with FakeTensorMode():
            tensor = torch.ones(3)
    else:
        tensor = torch.ones(3, device="meta")
    with (
        patch.object(jax.dlpack, "from_dlpack") as export,
        pytest.raises(TypeError, match="real tensors are required"),
    ):
        torch_to_jax(tensor)
    export.assert_not_called()


@pytest.mark.parametrize(
    "dtype", [torch.int64, torch.uint64, torch.float64, torch.complex128]
)
def test_torch_to_jax_rejects_implicit_64bit_narrowing(dtype):
    tensor = torch.ones(
        3, dtype=dtype, device=jax_to_torch_device(jax.local_devices()[0])
    )
    with (
        jax.enable_x64(False),
        pytest.raises(TypeError, match="64-bit.*JAX x64 disabled"),
    ):
        torch_to_jax(tensor)


def test_device_mapping_matches_dlpack():
    # Exercise CPU even when the test suite runs on a GPU.
    devices = set(jax.local_devices()) | set(jax.local_devices(backend="cpu"))
    for device in devices:
        array = jax.device_put(jnp.ones(2), device)
        tensor = torch.from_dlpack(array)
        assert jax_to_torch_device(device) == tensor.device


@pytest.mark.parametrize("accelerator_type", ["cuda", "xpu"])
def test_gpu_mapping_uses_local_hardware_index(accelerator_type):
    # Global JAX IDs need not match process-local hardware indices.
    device = Mock(platform="gpu", id=9, local_hardware_id=2)
    with patch.object(
        torch.accelerator,
        "current_accelerator",
        return_value=torch.device(accelerator_type),
    ):
        assert jax_to_torch_device(device) == torch.device(accelerator_type, 2)


def test_gpu_mapping_requires_available_torch_accelerator():
    device = Mock(platform="gpu", local_hardware_id=0)
    with (
        patch.object(torch.accelerator, "current_accelerator", return_value=None),
        pytest.raises(ValueError, match="No PyTorch accelerator"),
    ):
        jax_to_torch_device(device)


def test_device_mapping_rejects_unsupported_platform():
    with pytest.raises(ValueError, match="Unsupported JAX platform.*tpu"):
        jax_to_torch_device(Mock(platform="tpu"))


@pytest.mark.parametrize("fail", [False, True])
def test_rng_context_replays_cpu_and_device_draws_and_restores_process_rng(fail):
    device = jax_to_torch_device(jax.local_devices()[0])
    before = get_torch_rng_state(device)
    with fork_torch_rng(device, seed=17):
        saved = get_torch_rng_state(device)
        expected = (torch.rand(4), torch.rand(4, device=device))
    saved_copy = tuple(value.copy() if value is not None else None for value in saved)
    try:
        with fork_torch_rng(device, state=saved):
            actual = (torch.rand(4), torch.rand(4, device=device))
            for left, right in zip(expected, actual, strict=True):
                torch.testing.assert_close(left, right)
            if fail:
                raise RuntimeError("body failed")
    except RuntimeError as error:
        assert fail and str(error) == "body failed"
    for left, right in zip(before, get_torch_rng_state(device), strict=True):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(saved, saved_copy, strict=True):
        np.testing.assert_array_equal(left, right)
