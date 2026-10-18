"""Validate Jumanji YAML tasks and compare nested populations with serial runs."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

jumanji = pytest.importorskip("jumanji")
from jumanji import specs
from jumanji.environments.routing.sokoban.generator import ToyGenerator

from evorl.agent import RandomAgent
from evorl.envs import AutoresetMode, MultiAgentEnv, MultiDiscrete, create_env
from evorl.envs.env import get_episode_done
from evorl.envs.jumanji import create_jumanji_env, create_wrapped_jumanji_env
from evorl.rollout import rollout

pytestmark = pytest.mark.integration
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs/env/jumanji"
REPRESENTATIVES = {
    "Game2048-v1",
    "RubiksCube-v0",
    "LevelBasedForaging-v0",
    "SearchAndRescue-v0",
}


def pytest_generate_tests(metafunc):
    if "jumanji_config_path" not in metafunc.fixturenames:
        return
    cases = []
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        name = OmegaConf.load(path).env_name
        for mode in AutoresetMode:
            if (
                metafunc.config.getoption("--jumanji-catalog")
                or mode == AutoresetMode.ENVPOOL
                or name in REPRESENTATIVES
            ):
                cases.append((path, mode))
    metafunc.parametrize(
        ("jumanji_config_path", "mode"),
        cases,
        ids=[f"{path.relative_to(CONFIG_ROOT)}-{mode.value}" for path, mode in cases],
    )


def test_yaml_catalogue_matches_supported_tasks():
    paths = sorted(CONFIG_ROOT.rglob("*.yaml"))
    assert paths
    seen = set()
    for path in paths:
        config = OmegaConf.load(path)
        assert config.env_type == "jumanji"
        assert config.env_name in jumanji.registered_environments()
        assert config.env_name not in seen
        assert 0 < config.max_episode_steps < 2147483647
        assert "multi_to_single" not in config
        seen.add(config.env_name)
    # Default Sokoban needs an external level dataset; its in-memory generator is
    # tested below without presenting it as the default training configuration.
    assert seen == jumanji.registered_environments() - {"Sokoban-v0"}


@pytest.mark.parametrize("env_name", sorted(jumanji.registered_environments()))
def test_registered_spaces(env_name):
    kwargs = {"generator": ToyGenerator()} if env_name == "Sokoban-v0" else {}
    native = jumanji.make(env_name, **kwargs)
    env = create_jumanji_env(env_name, **kwargs)
    state = jax.jit(env.reset)(jax.random.key(1))
    if isinstance(env, MultiAgentEnv):
        spaces = [
            (env.action_space[agent], env.obs_space[agent], state.obs[agent])
            for agent in env.agents
        ]
    else:
        spaces = [(env.action_space, env.obs_space, state.obs)]
    for action_space, obs_space, observation in spaces:
        action = action_space.sample(jax.random.key(0))
        assert action_space.contains(action)
        assert obs_space.contains(observation)
        assert jax.tree.structure(observation) == jax.tree.structure(
            obs_space.shape, is_leaf=lambda x: isinstance(x, tuple)
        )
    if not isinstance(env, MultiAgentEnv):
        native.action_spec.validate(action)
        assert action.shape == native.action_spec.shape
        if isinstance(native.action_spec, specs.MultiDiscreteArray):
            assert isinstance(env.action_space, MultiDiscrete)


def test_configured_nested_vmap_matches_serial(jumanji_config_path, mode):
    config = OmegaConf.load(jumanji_config_path)
    env = create_env(
        config,
        parallel=2,
        episode_length=2,
        autoreset_mode=mode,
        record_ori_obs=True,
    )
    _check_nested_rollout(env)


@pytest.mark.parametrize("mode", list(AutoresetMode))
def test_sokoban_in_memory_levels(mode):
    env = create_wrapped_jumanji_env(
        "Sokoban-v0",
        generator=ToyGenerator(),
        parallel=2,
        episode_length=2,
        autoreset_mode=mode,
        record_ori_obs=True,
    )
    _check_nested_rollout(env)


def _check_nested_rollout(env):
    if isinstance(env.unwrapped, MultiAgentEnv):
        agent_state = None

        def action_fn(agent_state, sample_batch, key):
            actions = {
                agent: jax.vmap(space.sample)(jax.random.split(k, env.num_envs))
                for (agent, space), k in zip(
                    env.action_space.items(),
                    jax.random.split(key, len(env.agents)),
                    strict=True,
                )
            }
            return actions, {}
    else:
        agent = RandomAgent()
        agent_state = agent.init(env.obs_space, env.action_space, jax.random.key(0))
        action_fn = agent.compute_actions

    def collect(key):
        reset_key, rollout_key = jax.random.split(key)
        return rollout(
            env.step,
            action_fn,
            env.reset(reset_key),
            agent_state,
            rollout_key,
            5,
            env_extra_fields=(
                "steps",
                "termination",
                "truncation",
                "ori_obs",
                "autoreset",
            ),
        )

    keys = jax.random.split(jax.random.key(1), (2, 2))
    trajectory, state = jax.jit(jax.vmap(jax.vmap(collect)))(keys)
    jax.block_until_ready(state)
    for leaf in jax.tree.leaves((trajectory.rewards, trajectory.dones)):
        assert leaf.shape == (2, 2, 5, 2)
    action_spaces = env.action_space
    if isinstance(env.unwrapped, MultiAgentEnv):
        for agent, space in action_spaces.items():
            assert trajectory.actions[agent].shape == (2, 2, 5, 2, *space.shape)
    else:
        assert trajectory.actions.shape == (2, 2, 5, 2, *action_spaces.shape)
    for leaf in jax.tree.leaves((state.reward, state.done)):
        assert leaf.shape == (2, 2, 2)
    assert env.episode_length == 2
    assert jnp.all(trajectory.extras.env_extras.steps <= 2)
    assert jnp.all(jnp.any(get_episode_done(trajectory.dones), axis=2))
    for leaf in jax.tree.leaves(trajectory):
        assert jnp.isfinite(leaf).all()
    reference = jax.jit(collect)
    for outer in np.ndindex(2, 2):
        expected = reference(keys[outer])
        actual = jax.tree.map(lambda x, pos=outer: x[pos], (trajectory, state))
        assert jax.tree.structure(actual) == jax.tree.structure(expected)
        for a, b in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(
                    jax.random.key_data(a), jax.random.key_data(b)
                )
            elif jnp.issubdtype(a.dtype, jnp.inexact):
                np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
            else:
                np.testing.assert_array_equal(a, b)
