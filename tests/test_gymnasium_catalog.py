"""Exercise current Gymnasium YAML tasks and their declared array layouts."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from evorl.envs import Discrete, create_env

pytestmark = pytest.mark.integration
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs/env/gymnasium"
REPRESENTATIVES = {
    "ant-v5.yaml",
    "humanoid-v5.yaml",
    "inverteddoublependulum-v5.yaml",
    "reacher-v5.yaml",
    "pusher-v5.yaml",
    "carracing-v3.yaml",
    "bipedalwalker-v3.yaml",
    "lunarlander-v3.yaml",
    "lunarlandercontinuous-v3.yaml",
    "cartpole_v1.yaml",
    "pendulum_v1.yaml",
    "mountaincar_v0.yaml",
    "acrobot_v1.yaml",
    "pong-v5.yaml",
    "pong-ram-v5.yaml",
}


def pytest_generate_tests(metafunc):
    if "gymnasium_config_path" not in metafunc.fixturenames:
        return
    paths = sorted(CONFIG_ROOT.rglob("*.yaml"))
    # v2/v3 MuJoCo environments use the separate legacy mujoco-py backend.
    paths = [
        p for p in paths if p.parent.name != "mujoco" or p.stem.endswith(("-v4", "-v5"))
    ]
    if not metafunc.config.getoption("--host-env-catalog"):
        paths = [p for p in paths if p.name in REPRESENTATIVES]
    metafunc.parametrize(
        "gymnasium_config_path",
        paths,
        ids=[str(p.relative_to(CONFIG_ROOT)) for p in paths],
    )


@pytest.mark.parametrize("host_envs", ["gymnasium"], indirect=True)
def test_gymnasium_configured_task(host_envs, gymnasium_config_path, monkeypatch):
    family = gymnasium_config_path.parent.name
    if family == "mujoco":
        pytest.importorskip("mujoco")
        if gymnasium_config_path.name == "pusher-v4.yaml":
            import mujoco

            if int(mujoco.__version__.split(".")[0]) >= 3:
                pytest.xfail("Upstream Pusher-v4 requires MuJoCo < 3; use Pusher-v5")
    elif family == "box2d":
        pytest.importorskip("Box2D")
        monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
        monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    elif family == "ALE":
        pytest.importorskip("ale_py")

    _, environments = host_envs
    config = OmegaConf.load(gymnasium_config_path)
    env = create_env(config, parallel=1, episode_length=3)
    native = environments[0]
    observation_space = native.single_observation_space
    action_space = native.single_action_space
    actions = jnp.asarray(
        action_space.sample(),
        dtype=jnp.int32 if isinstance(env.action_space, Discrete) else jnp.float32,
    ).reshape((1, *action_space.shape))

    @jax.jit
    def run(key):
        state = env.reset(key)

        def step(state, _):
            state = env.step(state, actions)
            return state, state.reward

        return jax.lax.scan(step, state, None, length=5)

    state, rewards = run(jax.random.key(0))
    jax.block_until_ready(state)
    assert state.obs.shape == (1, *observation_space.shape)
    assert state.obs.dtype == jax.dtypes.canonicalize_dtype(observation_space.dtype)
    assert state.reward.shape == state.done.shape == (1,)
    assert rewards.shape == (5, 1)
    assert state.reward.dtype == state.done.dtype == jnp.float32
    assert np.isfinite(state.obs).all()
    assert np.isfinite(rewards).all()
