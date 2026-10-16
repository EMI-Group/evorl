"""Check explicit VMA casts without altering unrelated mesh axes."""

import os
import subprocess
import sys
import textwrap


def test_mark_varying_on_one_of_two_manual_axes():
    code = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from evorl.distributed.sharding import mark_varying

        mesh = jax.make_mesh((2, 2), ("DP", "POP"))
        with jax.set_mesh(mesh):
            @jax.shard_map(in_specs=jax.P("DP"), out_specs=jax.P("DP"))
            def accumulate(block):
                initial = jnp.zeros(block.shape)
                # None must stay a no-op even inside a Manual region.
                unchanged = mark_varying(initial, None)
                assert not jax.typeof(unchanged).manual_axis_type.varying
                carry = mark_varying({"total": unchanged, "block": block}, ("DP",))
                # Reapplying to already-varying leaves must also work.
                carry = mark_varying(carry, "DP")
                def step(carry, _):
                    return {"total": carry["total"] + carry["block"],
                            "block": carry["block"]}, ()
                final, _ = jax.lax.scan(step, carry, (), length=2)
                return final["total"]

            inputs = jax.device_put(jnp.arange(4, dtype=jnp.float32), jax.P("DP"))
            compiled = jax.jit(accumulate).lower(inputs).compile()
            np.testing.assert_array_equal(compiled(inputs), 2 * inputs)
            hlo = compiled.as_text().lower()
            assert "all-reduce" not in hlo and "all-gather" not in hlo
            # out_specs demands replication over POP; marking that axis would
            # fail strict VMA checking, so successful compilation checks scope.
        local = jnp.arange(2)
        assert mark_varying(local, None) is local
        assert mark_varying(local, ()) is local
        try:
            mark_varying(local)
        except TypeError:
            pass
        else:
            raise AssertionError("axis_name must be supplied explicitly")
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "JAX_NUM_CPU_DEVICES": "4",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            "CUDA_VISIBLE_DEVICES": "",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
