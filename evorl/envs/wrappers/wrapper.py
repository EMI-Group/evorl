import chex

from evorl.types import Action

from ..env import Env, EnvState, Space


class Wrapper(Env):
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env: Env):
        """Initialize the env wrapper.

        Args:
            env: the orginal env.
        """
        self.env = env

    def reset(self, key: chex.PRNGKey) -> EnvState:
        return self.env.reset(key)

    def step(self, state: EnvState, action: Action) -> EnvState:
        return self.env.step(state.env_state, action)

    @property
    def obs_space(self) -> Space:
        return self.env.obs_space

    @property
    def action_space(self) -> Space:
        return self.env.action_space

    @property
    def unwrapped(self) -> Env:
        if isinstance(self.env, Wrapper) and hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


def get_wrapper(env: Env, wrapper_cls: type) -> Wrapper | None:
    """Return a specific wrapper of an env."""
    if isinstance(env, wrapper_cls):
        return env
    elif hasattr(env, "env"):
        return get_wrapper(env.env, wrapper_cls)
    else:
        return None
