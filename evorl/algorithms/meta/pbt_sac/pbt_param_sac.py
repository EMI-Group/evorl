import chex
import jax

from evorl.types import PyTreeDict, State

from ..pbt_workflow import PBTOffpolicyWorkflowTemplate, PBTOptState
from ..pbt_utils import uniform_init, log_uniform_init


class PBTParamSACWorkflow(PBTOffpolicyWorkflowTemplate):
    @classmethod
    def name(cls):
        return "PBT-ParamSAC"

    def _setup_pop_and_pbt_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[chex.ArrayTree, PBTOptState]:
        search_space = self.config.search_space
        pop_size = self.config.pop_size

        def _init(hp, key):
            match hp:
                case "discount_g" | "log_alpha":
                    return uniform_init(search_space[hp], key, pop_size)
                case "actor_loss_weight" | "critic_loss_weight":
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
        agent_state = workflow_state.agent_state
        agent_state = agent_state.replace(
            params=agent_state.params.replace(
                log_alpha=hyperparams.log_alpha,
            ),
            extra_state=agent_state.extra_state.replace(
                discount_g=hyperparams.discount_g,
            ),
        )

        # make a shadow copy
        hyperparams = hyperparams.replace()
        hyperparams.pop("discount_g")
        hyperparams.pop("log_alpha")
        hp_state = workflow_state.hp_state.replace(**hyperparams)

        return workflow_state.replace(
            agent_state=agent_state,
            hp_state=hp_state,
        )
