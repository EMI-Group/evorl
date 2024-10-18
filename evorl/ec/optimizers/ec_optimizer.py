from abc import ABCMeta, abstractmethod

import chex

from evorl.types import PyTreeNode, PyTreeData, PyTreeDict

ECState = PyTreeData | PyTreeDict  # used for type hinting


class EvoOptimizer(PyTreeNode, metaclass=ABCMeta):
    """
    By default, all EvoOptimizer maximize the fitness.
    This is different from the behavior in EvoX.
    """

    @abstractmethod
    def init(self, *args, **kwargs) -> ECState:
        pass

    @abstractmethod
    def tell(
        self, state: ECState, xs: chex.ArrayTree, fitnesses: chex.Array
    ) -> ECState:
        "Update the optimizer state based on the fitnesses of the candidate solutions"
        pass

    @abstractmethod
    def ask(self, state: ECState) -> tuple[chex.ArrayTree, ECState]:
        "Generate new candidate solutions"
        pass
