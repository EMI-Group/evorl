import importlib

from .space import Space, Box, Discrete
from .env import Env, EnvState, EnvStepFn, EnvResetFn
from .multi_agent_env import MultiAgentEnv
from .wrappers.training_wrapper import AutoresetMode
from .brax import create_brax_env, create_wrapped_brax_env


# TODO: unifiy env creator
def create_env(env_cfg, **kwargs):
    """Unified env creator.

    Args:
        env_name: environment name
        env_type: env package name, eg: 'brax'
    """
    env_type = env_cfg.env_type
    env_name = env_cfg.env_name

    if env_type == "brax":
        env = create_wrapped_brax_env(env_name, **kwargs)
    elif env_type == "gymnax":
        env = create_wrapped_gymnax_env(env_name, **kwargs)
    elif env_type == "jumanji":
        env = create_jumanji_env(env_name, **kwargs)
    elif env_type == "jaxmarl":
        env = create_wrapped_mabrax_env(env_name, **kwargs)
    elif env_type == "envpool":
        if env_cfg.env_backend in ["gym", "gymnasium"]:
            env = create_envpool_gym_env(
                env_name, env_backend=env_cfg.env_backend, **kwargs
            )
        else:
            raise ValueError(f"env_backend {env_cfg.env_backend} not supported")
    else:
        raise ValueError(f"env_type {env_type} not supported")

    return env


__all__ = [
    "Env",
    "EnvState",
    "MultiAgentEnv",
    "Space",
    "Box",
    "Discrete",
    "AutoresetMode",
    "create_env",
    "create_brax_env",
    "create_wrapped_brax_env",
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
    from .envpool import create_envpool_gym_env

    __all__.extend(["create_envpool_gym_env"])
