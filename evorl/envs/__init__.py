import importlib

from .space import Space, Box, Discrete, MultiDiscrete, SpaceContainer, is_leaf_space
from .env import BaseEnv, Env, EnvState, EnvStepFn, EnvResetFn
from .multi_agent_env import MultiAgentEnv
from .wrappers.training_wrapper import AutoresetMode
from .brax import create_brax_env, create_wrapped_brax_env
from .mujoco_playground import (
    create_mujoco_playground_env,
    create_wrapped_mujoco_playground_env,
)
from .gymnasium import create_gymnasium_env
from .toy import create_wrapped_deep_sea_treasure


def create_env(env_cfg, **kwargs) -> BaseEnv:
    """Create an environment from configuration defaults and explicit overrides.

    Args:
        env_cfg: Configuration containing `env_name` and `env_type`. All other
            fields are forwarded to the environment factory, with
            `max_episode_steps` renamed to `episode_length`.
        **kwargs: Factory arguments overriding configuration defaults, including
            an explicit `None`. Use `episode_length` to override
            `env_cfg.max_episode_steps`.

    Returns:
        The created environment.
    """
    env_type = env_cfg.env_type
    env_name = env_cfg.env_name

    for name, value in env_cfg.items():
        if name not in ("env_name", "env_type", "max_episode_steps"):
            kwargs.setdefault(name, value)
    if "max_episode_steps" in env_cfg:
        kwargs.setdefault("episode_length", env_cfg.max_episode_steps)

    match env_type:
        case "toy":
            env = create_wrapped_deep_sea_treasure(env_name, **kwargs)
        case "brax":
            env = create_wrapped_brax_env(env_name, **kwargs)
        case "playground":
            env = create_wrapped_mujoco_playground_env(env_name, **kwargs)
        case "gymnax":
            env = create_wrapped_gymnax_env(env_name, **kwargs)
        case "jumanji":
            env = create_wrapped_jumanji_env(env_name, **kwargs)
        case "jaxmarl":
            env = create_wrapped_jaxmarl_env(env_name, **kwargs)
        case "envpool":
            env = create_envpool_env(env_name, **kwargs)
        case "gymnasium":
            env = create_gymnasium_env(env_name, **kwargs)
        case "mo_gymnasium":
            env = create_mo_gymnasium_env(env_name, **kwargs)
        case _:
            raise ValueError(f"env_type {env_type} not supported")

    return env


__all__ = [
    "BaseEnv",
    "Env",
    "EnvState",
    "MultiAgentEnv",
    "Space",
    "Box",
    "Discrete",
    "MultiDiscrete",
    "SpaceContainer",
    "AutoresetMode",
    "is_leaf_space",
    "create_env",
    "create_brax_env",
    "create_wrapped_brax_env",
    "create_mujoco_playground_env",
    "create_wrapped_mujoco_playground_env",
    "create_gymnasium_env",
]

if importlib.util.find_spec("gymnax") is not None:
    from .gymnax import create_gymnax_env, create_wrapped_gymnax_env

    __all__.extend(["create_gymnax_env", "create_wrapped_gymnax_env"])

if importlib.util.find_spec("jumanji") is not None:
    from .jumanji import create_jumanji_env, create_wrapped_jumanji_env

    __all__.extend(["create_jumanji_env", "create_wrapped_jumanji_env"])

if importlib.util.find_spec("jaxmarl") is not None:
    from .jaxmarl import create_jaxmarl_env, create_wrapped_jaxmarl_env

    __all__.extend(["create_jaxmarl_env", "create_wrapped_jaxmarl_env"])

if importlib.util.find_spec("envpool") is not None:
    from .envpool import create_envpool_env

    __all__.extend(["create_envpool_env"])

if importlib.util.find_spec("mo_gymnasium") is not None:
    from .mo_gymnasium import create_mo_gymnasium_env

    __all__.extend(["create_mo_gymnasium_env"])
