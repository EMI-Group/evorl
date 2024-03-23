import jax
from evox import Stateful, State

from evox.core.module import MetaStatefulModule

class WorkflowMetaStateful(MetaStatefulModule):
    def __new__(
        cls,
        name,
        bases,
        class_dict,
        force_wrap=["__call__"],
        ignore=["init", "setup", "enable_jit"],
        ignore_prefix="_",
    ):

        return super().__new__(cls, name, bases, class_dict, force_wrap, ignore, ignore_prefix)
        # return type.__new__(cls, name, bases, class_dict)

class Workflow(Stateful, metaclass=WorkflowMetaStateful):
    def step(self, state: State) -> State:
        raise NotImplementedError

    def learn(self, state: State) -> State:
        """
            run the complete learning process.
            Note: this is designed for the non pure function. Don't wrap it with jit.
        """
        raise NotImplementedError

    def enable_jit(self) -> None:
        self.step = jax.jit(self.step)
