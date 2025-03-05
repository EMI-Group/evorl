import chex
import jax

from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import tree_deepcopy

from ..pbt_workflow import PBTWorkflowTemplate, PBTOptState
from ..pbt_utils import log_uniform_init


class PBTParamOpenESWorkflow(PBTWorkflowTemplate):
    @classmethod
    def name(cls):
        return "PBT-ParamOpenES"

    def _setup_pop_and_pbt_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[chex.ArrayTree, PBTOptState]:
        search_space = self.config.search_space
        pop_size = self.config.pop_size

        def _init(hp, key):
            # "ec_noise_std" | "ec_lr"
            return log_uniform_init(search_space[hp], key, pop_size)

        pop = PyTreeDict(
            {
                hp: _init(hp, key)
                for hp, key in zip(
                    search_space.keys(), jax.random.split(key, len(search_space))
                )
            }
        )

        return pop, PBTOptState()

    def apply_hyperparams_to_workflow_state(
        self, workflow_state: State, hyperparams: PyTreeDict[str, chex.Numeric]
    ):
        ec_opt_state = workflow_state.ec_opt_state

        optax_opt_state = ec_opt_state.opt_state
        optax_opt_state = tree_deepcopy(optax_opt_state)
        optax_opt_state.hyperparams["learning_rate"] = hyperparams.ec_lr

        ec_opt_state = ec_opt_state.replace(
            opt_state=optax_opt_state,
            noise_std=hyperparams.ec_noise_std,
        )

        return workflow_state.replace(ec_opt_state=ec_opt_state)
