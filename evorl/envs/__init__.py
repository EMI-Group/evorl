from .env import Env, EnvState
from .space import Discrete, Box




from .brax import create_wrapped_brax_env

import importlib
if importlib.util.find_spec("gymnax") is not None:
    from .gymnax import create_wrapped_gymnax_env

if importlib.util.find_spec("jumanji") is not None:
    from .jumanji import create_jumanji_env

if importlib.util.find_spec("jaxmarl") is not None:
    from .jaxmarl import create_wrapped_mabrax_env


# TODO: unifiy env creator
def create_env(env_name: str, env_type: str,**kwargs):
    """
        Unified env creator.

        Args:
            env_name: environment name
            env_type: env package name, eg: 'brax'
    """
    if env_type == 'brax':
        env = create_wrapped_brax_env(env_name, **kwargs)
    elif env_type == 'gymnax':
        env = create_wrapped_gymnax_env(env_name, **kwargs)
    elif env_type == 'jumanji':
        env = create_jumanji_env(env_name, **kwargs)
    elif env_type == 'jaxmarl':
        env = create_wrapped_mabrax_env(env_name, **kwargs)
    else:
        raise ValueError(f'env_type {env_type} not supported')

    # if autoreset:
    #     env = EpisodeWrapper(env, episode_length,
    #                          record_episode_return=True, discount=discount)
    #     if fast_reset:
    #         env = VmapAutoResetWrapperV2(env, num_envs=parallel)
    #     else:
    #         env = VmapAutoResetWrapper(env, num_envs=parallel)
    # else:
    #     env = OneEpisodeWrapper(env, episode_length)
    #     env = VmapWrapper(env, num_envs=parallel, vmap_step=False)

    return env