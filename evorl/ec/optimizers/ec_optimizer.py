from abc import ABCMeta, abstractmethod

import chex

from evorl.types import PyTreeNode, PyTreeData, PyTreeDict

ECState = PyTreeData | PyTreeDict  # used for type hinting


class EvoOptimizer(PyTreeNode, metaclass=ABCMeta):
    """Evolutionary optimizer interface; all fitness objectives are maximized."""

    @abstractmethod
    def init(self, *args, **kwargs) -> ECState:
        raise NotImplementedError

    @abstractmethod
    def ask(self, state: ECState) -> tuple[chex.ArrayTree, ECState]:
        """Generate new candidate solutions."""
        raise NotImplementedError

    @abstractmethod
    def tell(
        self, state: ECState, fitnesses: chex.ArrayTree
    ) -> tuple[PyTreeDict, ECState]:
        """Update the optimizer state based on the fitnesses of the candidate solutions.

        Args:
            state: The current optimizer state
            fitnesses: The fitnesses of the candidate solutions
        """
        raise NotImplementedError
