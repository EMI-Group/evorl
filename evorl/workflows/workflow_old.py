import jax
import chex
from evox import Stateful, State

from evox.core.module import MetaStatefulModule

class WorkflowMetaStateful(MetaStatefulModule):
    def __new__(
        cls,
        name,
        bases,
        class_dict
    ):
        # return super().__new__(cls, name, bases, class_dict, 
        #                        ignore=["init", "setup", "enable_jit","enable_multi_devices"])
        return type.__new__(cls, name, bases, class_dict)

class Workflow(Stateful, metaclass=WorkflowMetaStateful):
    def setup(self, key: jax.Array) -> State:
        """
            custom setup.
            When call public API init(), setup() would be recursively called.
        """
        raise NotImplementedError

    def step(self, key: chex.PRNGKey) -> State:
        raise NotImplementedError

    def learn(self, state: State) -> State:
        """
            run the complete learning process.
            Note: this is designed for the non pure function. Don't wrap it with jit.
        """
        raise NotImplementedError

    @classmethod
    def enable_jit(cls) -> None:
        """
        in-place update Workflow class with jitted functions        
        """
        cls.step = jax.jit(cls.step, static_argnums=(0,))
