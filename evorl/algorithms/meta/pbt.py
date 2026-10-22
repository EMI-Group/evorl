import logging

import jax
import chex
import optax
from flax import nnx

from evorl.types import PyTreeDict, State

from .pbt_workflow import PBTWorkflowTemplate, PBTOptState
from .pbt_utils import log_uniform_init

logger = logging.getLogger(__name__)


class PBTWorkflow(PBTWorkflowTemplate):
    """A minimal Example of PBT that tunes the lr of PPO."""

    @classmethod
    def name(cls):
        return "PBT"

    def _customize_optimizer(self) -> None:
        """Customize the target workflow's optimizer."""
        self.workflow.optimizer = optax.inject_hyperparams(
            optax.adam, static_args=("b1", "b2", "eps", "eps_root")
        )(learning_rate=self.config.search_space.lr.low)

    def _setup_pop_and_pbt_optimizer(
        self, key: jax.Array
    ) -> tuple[chex.ArrayTree, PBTOptState]:
        pop = PyTreeDict(
            lr=log_uniform_init(self.config.search_space.lr, key, self.config.pop_size)
        )

        return pop, PBTOptState()

    def apply_hyperparams_to_workflow_state(
        self, workflow_state: State, hyperparams: PyTreeDict[str, jax.typing.ArrayLike]
    ) -> State:
        optimizer = nnx.merge(workflow_state.opt_state, copy=True)
        optimizer.opt_state.hyperparams["learning_rate"][...] = hyperparams.lr
        return workflow_state.replace(opt_state=nnx.unpack(optimizer))
