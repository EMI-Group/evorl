import jax
import chex
from evox import Stateful, State

from evox.core.module import MetaStatefulModule

from typing import Any, Tuple, Union

class Workflow:
    """
        A duck-type of evox.Workflow
    """

    def setup(self, key: jax.Array) -> State:
        """
            Custom setup.
            When call public API init(), setup() would be recursively called.
        """
        raise NotImplementedError

    def step(self, key: chex.PRNGKey) -> Union[State, Tuple[State, Any]]:
        raise NotImplementedError

    def learn(self, state: State) -> State:
        """
            run the complete learning process.
            Note: this is designed for the non pure function. Don't wrap it with jit.
        """
        raise NotImplementedError

    def init(self, key: jax.Array) -> State:
        """
            Initialize the state of the module.
            This is the public API to call for instance state initialization.
        """
        return self.setup(key)

    @classmethod
    def enable_jit(cls) -> None:
        """
        in-place update Workflow class with jitted functions        
        """
        cls.step = jax.jit(cls.step, static_argnums=(0,))
