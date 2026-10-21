"""Environment configuration forwarding and explicit override precedence."""

from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from evorl import envs


@pytest.mark.parametrize(
    "backend,factory_name,field,default,override",
    [
        ("brax", "create_wrapped_brax_env", "backend", "spring", "generalized"),
        (
            "playground",
            "create_wrapped_mujoco_playground_env",
            "config_overrides",
            {"impl": "jax"},
            {"impl": "warp"},
        ),
        (
            "gymnax",
            "create_wrapped_gymnax_env",
            "env_kwargs",
            {"size": 4},
            {"size": 8},
        ),
        ("jumanji", "create_wrapped_jumanji_env", "time_limit", 100, 200),
        (
            "jaxmarl",
            "create_wrapped_jaxmarl_env",
            "layout",
            "forced_coord",
            "cramped_room",
        ),
        ("envpool", "create_envpool_env", "num_threads", 2, 4),
        ("gymnasium", "create_gymnasium_env", "obs_type", "rgb", "ram"),
    ],
)
@pytest.mark.parametrize("explicit", [False, True], ids=["yaml", "override"])
def test_factory_configuration_and_override_precedence(
    monkeypatch, backend, factory_name, field, default, override, explicit
):
    factory = Mock()
    monkeypatch.setattr(envs, factory_name, factory, raising=False)
    config = OmegaConf.create(
        dict(env_type=backend, env_name="task", max_episode_steps=300)
    )
    config[field] = default
    original_config = OmegaConf.to_container(config)
    OmegaConf.set_readonly(config, True)
    kwargs = dict(parallel=2)
    if explicit:
        kwargs.update(episode_length=500, **{field: override})
    original_kwargs = kwargs.copy()

    assert envs.create_env(config, **kwargs) is factory.return_value

    factory.assert_called_once_with(
        "task",
        parallel=2,
        episode_length=500 if explicit else 300,
        **{field: override if explicit else default},
    )
    assert OmegaConf.to_container(config) == original_config
    assert kwargs == original_kwargs


def test_explicit_none_overrides_configured_reward_keys(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(envs, "create_wrapped_brax_env", factory)
    config = OmegaConf.create(
        dict(env_type="brax", env_name="ant", reward_keys=["reward_forward"])
    )

    envs.create_env(config, reward_keys=None)

    factory.assert_called_once_with("ant", reward_keys=None)
