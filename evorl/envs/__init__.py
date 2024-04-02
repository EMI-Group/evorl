from .env import Env, EnvState
from .space import Discrete, Box

from .wrappers.training_wrapper import EpisodeWrapper, OneEpisodeWrapper, VmapAutoResetWrapper, VmapWrapper, VmapAutoResetWrapperV2


from .brax import create_brax_env

import importlib
if importlib.util.find_spec("gymnax") is not None:
    from .gymnax import create_gymnax_env

if importlib.util.find_spec("jumanji") is not None:
    from .jumanji import create_jumanji_env


# TODO: unifiy env creator
def create_env(env_name: str, env_type: str,
               episode_length: int = 1000,
               parallel: int = 1,
               autoreset: bool = True,
               fast_reset: bool = False,
               discount: float = 1.0,
               **kwargs):
    """
        Unified env creator.

        Args:
            env_name: environment name
            env_hint: env package name, eg: 'brax'
    """
    if env_type == 'brax':
        env = create_brax_env(env_name, **kwargs)
    elif env_type == 'gymnax':
        env = create_gymnax_env(env_name, **kwargs)
    elif env_type == 'jumanji':
        env = create_jumanji_env(env_name, **kwargs)
    else:
        raise ValueError(f'env_type {env_type} not supported')

    if autoreset:
        env = EpisodeWrapper(env, episode_length,
                             record_episode_return=True, discount=discount)
        if fast_reset:
            env = VmapAutoResetWrapperV2(env, num_envs=parallel)
        else:
            env = VmapAutoResetWrapper(env, num_envs=parallel)
    else:
        env = OneEpisodeWrapper(env, episode_length)
        env = VmapWrapper(env, num_envs=parallel, vmap_step=False)

    return env