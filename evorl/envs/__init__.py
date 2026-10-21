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


def create_env(env_cfg, **kwargs) -> BaseEnv:
    """Unified env creator.

    Args:
        env_cfg: The environment configuration. Gymnasium configurations can set
            `obs_type` to select the Atari observation format. Gymnax
            and JaxMARL configurations can set `env_kwargs` for environment
            construction.
            Jumanji multi-agent tasks use the `MultiAgentEnv` API automatically.
            Brax and Playground configurations can set `reward_keys` to expose
            selected per-step metrics as a multi-objective reward vector.
        **kwargs: Additional keyword arguments for the environment creator.
            Explicit `obs_type` and `env_kwargs` arguments override the
            corresponding configuration values.

    Returns:
        The created environment.
    """
    env_type = env_cfg.env_type
    env_name = env_cfg.env_name

    match env_type:
        case "brax":
            if "reward_keys" in env_cfg:
                kwargs.setdefault("reward_keys", env_cfg.reward_keys)
            env = create_wrapped_brax_env(env_name, **kwargs)
        case "playground":
            if "reward_keys" in env_cfg:
                kwargs.setdefault("reward_keys", env_cfg.reward_keys)
            env = create_wrapped_mujoco_playground_env(env_name, **kwargs)
        case "gymnax":
            if "env_kwargs" in env_cfg:
                kwargs.setdefault("env_kwargs", env_cfg.env_kwargs)
            env = create_wrapped_gymnax_env(env_name, **kwargs)
        case "jumanji":
            env = create_wrapped_jumanji_env(env_name, **kwargs)
        case "jaxmarl":
            if "env_kwargs" in env_cfg:
                kwargs.setdefault("env_kwargs", env_cfg.env_kwargs)
            env = create_wrapped_jaxmarl_env(env_name, **kwargs)
        case "envpool":
            env = create_envpool_env(env_name, **kwargs)
        case "gymnasium":
            if "obs_type" in env_cfg:
                kwargs.setdefault("obs_type", env_cfg.obs_type)
            env = create_gymnasium_env(env_name, **kwargs)
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
