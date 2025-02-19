import importlib

from .brax import create_brax_env, create_wrapped_brax_env
from .env import Env, EnvState, EnvStepFn, EnvResetFn
from .multi_agent_env import MultiAgentEnv
from .space import Space, Box, Discrete
from .wrappers.wrapper import get_wrapper
from .wrappers.training_wrapper import AutoresetMode

if importlib.util.find_spec("gymnax") is not None:
    from .gymnax import create_gymnax_env, create_wrapped_gymnax_env

if importlib.util.find_spec("jumanji") is not None:
    from .jumanji import create_jumanji_env

if importlib.util.find_spec("jaxmarl") is not None:
    from .jaxmarl import create_mabrax_env, create_wrapped_mabrax_env

if importlib.util.find_spec("envpool") is not None:
    from .envpool import creat_gym_env


# TODO: unifiy env creator
def create_env(env_name: str, env_type: str, **kwargs):
    """Unified env creator.

    Args:
        env_name: environment name
        env_type: env package name, eg: 'brax'
    """
    if env_type == "brax":
        env = create_wrapped_brax_env(env_name, **kwargs)
    elif env_type == "gymnax":
        env = create_wrapped_gymnax_env(env_name, **kwargs)
    elif env_type == "jumanji":
        env = create_jumanji_env(env_name, **kwargs)
    elif env_type == "jaxmarl":
        env = create_wrapped_mabrax_env(env_name, **kwargs)
    elif env_type.startswith("envpool"):
        _env_type = env_type.split("_")[1]
        if _env_type in ["gym", "gymnasium"]:
            env = creat_gym_env(
                env_name, gymnasium_env=(_env_type == "gymnasium"), **kwargs
            )
        else:
            raise ValueError(f"env_type {env_type} not supported")
    else:
        raise ValueError(f"env_type {env_type} not supported")

    return env


__all__ = [
    "Env",
    "EnvState",
    "EnvStepFn",
    "EnvResetFn",
    "MultiAgentEnv",
    "Space",
    "Box",
    "Discrete",
    "AutoresetMode",
    "get_wrapper",
    "create_env",
    "create_brax_env",
    "create_wrapped_brax_env",
    "create_gymnax_env",
    "create_wrapped_gymnax_env",
    "create_jumanji_env",
    "create_mabrax_env",
    "create_wrapped_mabrax_env",
    "creat_gym_env",
]
