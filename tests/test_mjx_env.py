import pytest
import jax
import jax.numpy as jnp

from evorl.envs import (
    create_wrapped_mujoco_playground_env,
    create_mujoco_playground_env,
    AutoresetMode,
    Box,
)
from evorl.envs.space import SpaceContainer
from evorl.envs.wrappers import get_wrapper
from evorl.envs.wrappers.training_wrapper import (
    EpisodeWrapper,
    OneEpisodeWrapper,
    VmapWrapper,
    VmapAutoResetWrapper,
    FastVmapAutoResetWrapper,
    VmapEnvPoolAutoResetWrapper,
)
from tests.utils import requires_gpu

DM_CONTROL_ENVS = ["CartpoleSwingup", "CheetahRun"]
LOCOMOTION_ENVS = ["Go1JoystickFlatTerrain", "BerkeleyHumanoidJoystickFlatTerrain"]
MANIPULATION_ENVS = ["PandaPickCube", "AlohaHandOver"]
ALL_TEST_ENVS = DM_CONTROL_ENVS + LOCOMOTION_ENVS + MANIPULATION_ENVS


def _get_obs_shape(state_obs, key="state"):
    """Get shape from obs, handling both array and dict obs."""
    if isinstance(state_obs, dict):
        return state_obs[key].shape
    return state_obs.shape


def _assert_obs_batched(state_obs, num_envs, obs_space):
    """Assert obs has correct batch dimensions."""
    if isinstance(obs_space, SpaceContainer):
        for key in obs_space.shape:
            assert state_obs[key].shape == (num_envs, *obs_space.shape[key])
    else:
        assert state_obs.shape == (num_envs, *obs_space.shape)


# ─── env creation ─────────────────────────────────────────────

class TestMjxEnvCreation:
    """Test basic env creation and space properties."""

    @pytest.mark.parametrize("env_name", ALL_TEST_ENVS)
    @pytest.mark.parametrize("impl", ["jax", "warp"])
    def test_env_creation(self, env_name, impl):
        if impl == "warp" and jax.default_backend() == "cpu":
            pytest.skip("Warp backend requires GPU/CUDA")

        env = create_mujoco_playground_env(
            env_name, config_overrides={"impl": impl}
        )
        assert isinstance(env.action_space, Box)
        assert hasattr(env, "obs_space")
        assert env.action_space.shape[0] > 0


# ─── _impl stripping ─────────────────────────────────────────

class TestImplStripping:
    """Verify _impl is correctly stripped from state to save memory."""

    @pytest.mark.parametrize(
        "env_name", ["CartpoleSwingup", "Go1JoystickFlatTerrain", "PandaPickCube"]
    )
    @pytest.mark.parametrize("impl", ["jax", "warp"])
    def test_impl_stripped(self, env_name, impl):
        if impl == "warp" and jax.default_backend() == "cpu":
            pytest.skip("Warp backend requires GPU/CUDA")

        env = create_mujoco_playground_env(env_name, config_overrides={"impl": impl})
        state = env.reset(jax.random.PRNGKey(0))
        assert state.env_state.data._impl is None

        action = jnp.zeros(env.action_space.shape)
        state = env.step(state, action)
        assert state.env_state.data._impl is None

    @pytest.mark.parametrize(
        "env_name", ["CartpoleSwingup", "Go1JoystickFlatTerrain", "PandaPickCube"]
    )
    @pytest.mark.parametrize("impl", ["jax", "warp"])
    def test_step_correctness_with_stripped_impl(self, env_name, impl):
        """Verify that physics results are identical regardless of _impl source."""
        if impl == "warp" and jax.default_backend() == "cpu":
            pytest.skip("Warp backend requires GPU/CUDA")

        env = create_mujoco_playground_env(env_name, config_overrides={"impl": impl})
        state = env.reset(jax.random.PRNGKey(1))
        action = jnp.zeros(env.action_space.shape)

        # Two consecutive steps from same state
        s1 = env.step(state, action)
        s2 = env.step(state, action)

        assert jnp.allclose(s1.env_state.data.qpos, s2.env_state.data.qpos)
        assert jnp.allclose(s1.env_state.data.qvel, s2.env_state.data.qvel)


# ─── wrapper modes ───────────────────────────────────────────

class TestWrappedEnvStep:
    """Test all autoreset modes across different environments."""

    NUM_ENVS = 4

    @pytest.mark.parametrize("env_name", ALL_TEST_ENVS)
    @pytest.mark.parametrize("impl", ["jax", "warp"])
    @pytest.mark.parametrize(
        "mode",
        [
            AutoresetMode.NORMAL,
            AutoresetMode.FAST,
            AutoresetMode.ENVPOOL,
            AutoresetMode.DISABLED,
        ],
    )
    def test_step(self, env_name, impl, mode):
        if impl == "warp" and jax.default_backend() == "cpu":
            pytest.skip("Warp backend requires GPU/CUDA")

        env = create_wrapped_mujoco_playground_env(
            env_name,
            episode_length=100,
            parallel=self.NUM_ENVS,
            autoreset_mode=mode,
            config_overrides={"impl": impl},
        )
        base_env = create_mujoco_playground_env(
            env_name, config_overrides={"impl": impl}
        )

        rng = jax.random.PRNGKey(42)
        state = env.reset(rng)

        _assert_obs_batched(state.obs, self.NUM_ENVS, base_env.obs_space)

        actions = jnp.zeros((self.NUM_ENVS, base_env.action_space.shape[0]))
        for _ in range(3):
            state = env.step(state, actions)

        _assert_obs_batched(state.obs, self.NUM_ENVS, base_env.obs_space)
        assert state.reward.shape == (self.NUM_ENVS,)
        assert state.done.shape == (self.NUM_ENVS,)
        assert state.env_state.data._impl is None


# ─── long rollout with lax.scan ──────────────────────────────

@requires_gpu
class TestLongRollout:
    """Test multi-step rollouts with jax.lax.scan to verify JIT stability."""

    NUM_ENVS = 4
    ROLLOUT_LEN = 20

    @pytest.mark.parametrize(
        "env_name", ["CartpoleSwingup", "Go1JoystickFlatTerrain", "PandaPickCube"]
    )
    @pytest.mark.parametrize("mode", [AutoresetMode.NORMAL, AutoresetMode.FAST])
    def test_scan_rollout(self, env_name, mode):
        # Default backend is warp on GPU
        env = create_wrapped_mujoco_playground_env(
            env_name,
            episode_length=100,
            parallel=self.NUM_ENVS,
            autoreset_mode=mode,
        )
        base_env = create_mujoco_playground_env(env_name)

        def rollout_step(carry, _):
            state = carry
            action = jnp.zeros((self.NUM_ENVS, base_env.action_space.shape[0]))
            state = env.step(state, action)
            return state, state.done

        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)
        final_state, dones = jax.lax.scan(
            rollout_step, state, None, length=self.ROLLOUT_LEN
        )

        assert dones.shape == (self.ROLLOUT_LEN, self.NUM_ENVS)
        _assert_obs_batched(final_state.obs, self.NUM_ENVS, base_env.obs_space)


# ─── wrapper structure ────────────────────────────────────────

@requires_gpu
class TestWrapperStructure:
    """Verify correct wrapper stacking for each autoreset mode."""

    ENV_NAME = "CartpoleSwingup"
    NUM_ENVS = 3

    def _has_wrapper(self, env, cls):
        return get_wrapper(env, cls) is not None

    def test_normal_mode(self):
        env = create_wrapped_mujoco_playground_env(
            self.ENV_NAME, parallel=self.NUM_ENVS, autoreset_mode=AutoresetMode.NORMAL
        )
        assert self._has_wrapper(env, EpisodeWrapper)
        assert self._has_wrapper(env, VmapAutoResetWrapper)

    def test_fast_mode(self):
        env = create_wrapped_mujoco_playground_env(
            self.ENV_NAME, parallel=self.NUM_ENVS, autoreset_mode=AutoresetMode.FAST
        )
        assert self._has_wrapper(env, EpisodeWrapper)
        assert self._has_wrapper(env, FastVmapAutoResetWrapper)

    def test_envpool_mode(self):
        env = create_wrapped_mujoco_playground_env(
            self.ENV_NAME, parallel=self.NUM_ENVS, autoreset_mode=AutoresetMode.ENVPOOL
        )
        assert self._has_wrapper(env, EpisodeWrapper)
        assert self._has_wrapper(env, VmapEnvPoolAutoResetWrapper)

    def test_disabled_mode(self):
        env = create_wrapped_mujoco_playground_env(
            self.ENV_NAME, parallel=self.NUM_ENVS, autoreset_mode=AutoresetMode.DISABLED
        )
        assert self._has_wrapper(env, OneEpisodeWrapper)
        assert self._has_wrapper(env, VmapWrapper)
