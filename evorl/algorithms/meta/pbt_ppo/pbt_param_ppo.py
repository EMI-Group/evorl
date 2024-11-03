import chex
import jax
from evorl.types import PyTreeDict, State

from ..pbt import PBTWorkflowTemplate


class PBTParamPPOWorkflow(PBTWorkflowTemplate):
    @classmethod
    def name(cls):
        return "PBT-ParamPPO"

    def _setup_pop(self, key: chex.PRNGKey) -> chex.ArrayTree:
        def _uniform_init(key, search_space):
            return jax.random.uniform(
                key,
                (self.config.pop_size,),
                minval=search_space.low,
                maxval=search_space.high,
            )

        search_space = self.config.search_space
        pop = PyTreeDict(
            {
                hp: _uniform_init(key, search_space[hp])
                for hp, key in zip(
                    search_space.keys(), jax.random.split(key, len(search_space))
                )
            }
        )

        return pop

    def apply_hyperparams_to_workflow_state(
        self, workflow_state: State, hyperparams: PyTreeDict[str, chex.Numeric]
    ):
        agent_state = workflow_state.agent_state
        agent_state = agent_state.replace(
            extra_state=agent_state.extra_state.replace(
                clip_epsilon=hyperparams.clip_epsilon
            )
        )

        # make a shadow copy
        hyperparams = hyperparams.replace()
        hyperparams.pop("clip_epsilon")

        return workflow_state.replace(
            agent_state=agent_state,
            hp_state=hyperparams,
        )
