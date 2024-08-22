from evorl.envs import (
    create_wrapped_brax_env,
    AutoresetMode,
    Box,
)
from evorl.envs.wrappers import get_wrapper
from evorl.envs.wrappers.training_wrapper import (
    EpisodeWrapper,
    OneEpisodeWrapper,
    VmapWrapper,
    VmapAutoResetWrapper,
    FastVmapAutoResetWrapper,
    VmapEnvPoolAutoResetWrapper,
)


def has_wrapper(env, wrapper_cls):
    return get_wrapper(env, wrapper_cls) is not None


def test_brax_env():
    num_envs = 7

    env = create_wrapped_brax_env("ant", parallel=num_envs)
    action_space = env.action_space
    obs_space = env.obs_space
    assert env.num_envs == num_envs
    assert isinstance(action_space, Box)
    assert isinstance(obs_space, Box)
    assert action_space.shape == (8,), action_space.shape
    assert obs_space.shape == (27,), obs_space.shape
    assert env.num_envs == num_envs


def test_has_brax_wrapper():
    num_envs = 3
    env = create_wrapped_brax_env(
        "ant", parallel=num_envs, autoreset_mode=AutoresetMode.NORMAL
    )

    assert has_wrapper(env, EpisodeWrapper)
    assert has_wrapper(env, VmapAutoResetWrapper)

    env = create_wrapped_brax_env(
        "ant", parallel=num_envs, autoreset_mode=AutoresetMode.FAST
    )
    assert has_wrapper(env, EpisodeWrapper)
    assert has_wrapper(env, FastVmapAutoResetWrapper)

    env = create_wrapped_brax_env(
        "ant", parallel=num_envs, autoreset_mode=AutoresetMode.ENVPOOL
    )
    assert has_wrapper(env, EpisodeWrapper)
    assert has_wrapper(env, VmapEnvPoolAutoResetWrapper)

    env = create_wrapped_brax_env(
        "ant", parallel=num_envs, autoreset_mode=AutoresetMode.DISABLED
    )
    assert has_wrapper(env, OneEpisodeWrapper)
    assert has_wrapper(env, VmapWrapper)
