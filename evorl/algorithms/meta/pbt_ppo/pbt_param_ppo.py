import chex
import jax
from omegaconf import OmegaConf

from evorl.types import PyTreeDict, State

from ..pbt import PBTWorkflowTemplate
from ..pbt_utils import uniform_init, log_uniform_init


class PBTParamPPOWorkflow(PBTWorkflowTemplate):
    @classmethod
    def name(cls):
        return "PBT-ParamPPO"

    def _setup_pop(self, key: chex.PRNGKey) -> chex.ArrayTree:
        search_space = self.config.search_space

        def _init(hp, key):
            match hp:
                case "actor_loss_weight" | "critic_loss_weight" | "clip_epsilon":
                    return log_uniform_init(search_space[hp], key, self.config.pop_size)
                case "entropy_loss_weight":
                    return -log_uniform_init(
                        OmegaConf.create(
                            dict(low=-search_space[hp].high, high=-search_space[hp].low)
                        ),
                        key,
                        self.config.pop_size,
                    )
                case "discount_g" | "gae_lambda_g":
                    return uniform_init(search_space[hp], key, self.config.pop_size)

        pop = PyTreeDict(
            {
                hp: _init(hp, key)
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
