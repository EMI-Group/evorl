from abc import ABCMeta, abstractmethod

import chex

from evorl.types import PyTreeNode, PyTreeData, PyTreeDict

ECState = PyTreeData | PyTreeDict  # used for type hinting


class EvoOptimizer(PyTreeNode, metaclass=ABCMeta):
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
    def ask(self, state: ECState, key: chex.PRNGKey) -> chex.ArrayTree:
        "Generate new candidate solutions"
        pass
