import jax

from ..env import BaseEnv, EnvState


class Wrapper[ActionT, SpaceT](BaseEnv[ActionT, SpaceT]):
    """Wrap a single-agent or multi-agent environment with the common API.

    Public attributes not defined by the wrapper are forwarded to the wrapped
    environment, including `agents` for multi-agent environments.
    Actions and spaces retain the types of the wrapped environment.
    """

    def __init__(self, env: BaseEnv[ActionT, SpaceT]):
        """Initialize the env wrapper.

        Args:
            env: The environment or another wrapper.
        """
        self.env = env

    def reset(self, key: jax.Array) -> EnvState:
        return self.env.reset(key)

    def step(self, state: EnvState, action: ActionT) -> EnvState:
        return self.env.step(state, action)

    @property
    def obs_space(self) -> SpaceT:
        return self.env.obs_space

    @property
    def action_space(self) -> SpaceT:
        return self.env.action_space

    @property
    def unwrapped(self) -> BaseEnv:
        """Return the EvoRL environment underneath all EvoRL wrappers."""
        if isinstance(self.env, Wrapper):
            return self.env.unwrapped
        else:
            return self.env

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


def get_wrapper[WrapperT: Wrapper](
    env: BaseEnv, wrapper_cls: type[WrapperT]
) -> WrapperT | None:
    """Return a specific wrapper of an env."""
    if isinstance(env, wrapper_cls):
        return env
    elif hasattr(env, "env"):
        return get_wrapper(env.env, wrapper_cls)
    else:
        return None
