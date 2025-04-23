import importlib

from .space import Space, Box, Discrete, SpaceContainer, is_leaf_space
from .env import Env, EnvState, EnvStepFn, EnvResetFn
from .multi_agent_env import MultiAgentEnv
from .wrappers.training_wrapper import AutoresetMode
from .brax import create_brax_env, create_wrapped_brax_env
from .gymnasium import create_gymnasium_env


def create_env(env_cfg, **kwargs) -> Env:
    """Unified env creator.

    Args:
        env_name: environment name
        env_type: env package name, eg: 'brax'
    """
    env_type = env_cfg.env_type
    env_name = env_cfg.env_name

    match env_type:
        case "brax":
            env = create_wrapped_brax_env(env_name, **kwargs)
        case "playground":
            env = create_wrapped_mujoco_playground_env(env_name, **kwargs)
        case "gymnax":
            env = create_wrapped_gymnax_env(env_name, **kwargs)
        case "jumanji":
            env = create_jumanji_env(env_name, **kwargs)
        case "jaxmarl":
            env = create_mabrax_env(env_name, **kwargs)
        case "envpool":
            if env_cfg.env_backend in ["gym", "gymnasium"]:
                env = create_envpool_env(
                    env_name, env_backend=env_cfg.env_backend, **kwargs
                )
            else:
                raise ValueError(f"env_backend {env_cfg.env_backend} not supported")
        case "gymnasium":
            env = create_gymnasium_env(env_name, **kwargs)
        case _:
            raise ValueError(f"env_type {env_type} not supported")

    return env


__all__ = [
    "Env",
    "EnvState",
    "MultiAgentEnv",
    "Space",
    "Box",
    "Discrete",
    "SpaceContainer",
    "AutoresetMode",
    "is_leaf_space",
    "create_env",
    "create_brax_env",
    "create_wrapped_brax_env",
    "create_gymnasium_env",
]

if importlib.util.find_spec("gymnax") is not None:
    from .gymnax import create_gymnax_env, create_wrapped_gymnax_env

    __all__.extend(["create_gymnax_env", "create_wrapped_gymnax_env"])

if importlib.util.find_spec("jumanji") is not None:
    from .jumanji import create_jumanji_env

    __all__.extend(["create_jumanji_env"])

if importlib.util.find_spec("jaxmarl") is not None:
    from .jaxmarl import create_mabrax_env, create_wrapped_mabrax_env

    __all__.extend(["create_mabrax_env", "create_wrapped_mabrax_env"])

if importlib.util.find_spec("envpool") is not None:
    from .envpool import create_envpool_env

    __all__.extend(["create_envpool_env"])

if importlib.util.find_spec("mujoco_playground") is not None:
    from .mujoco_playground import (
        create_mujoco_playground_env,
        create_wrapped_mujoco_playground_env,
    )

    __all__.extend(
        ["create_mujoco_playground_env", "create_wrapped_mujoco_playground_env"]
    )
