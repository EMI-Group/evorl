from .env import Env
from .space import Discrete, Box

from .brax import BraxEnvAdapter
from .brax import create_env as create_brax_env


# TODO: unifiy env creator
def create_env(env_name: str, env_type: str, *args, **kwargs):
    """
        Unified env creator.

        Args:
            env_name: environment name
            env_hint: env package name, eg: 'brax'
    """
    if env_type == 'brax':
        return create_brax_env(env_name, *args, **kwargs)
    else:
        raise ValueError(f'env_type {env_type} not supported')
