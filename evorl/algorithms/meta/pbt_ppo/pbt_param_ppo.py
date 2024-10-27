import chex
import jax
from evorl.distributed import (
    POP_AXIS_NAME,
    tree_device_put,
)
from evorl.types import PyTreeDict, State

from ..pbt import PBTWorkflow, PBTWorkflowMetric


class PBTParamPPOWorkflow(PBTWorkflow):
    @classmethod
    def name(cls):
        return "PBT-ParamPPO"

    def setup(self, key: chex.PRNGKey):
        pop_size = self.config.pop_size

        key, workflow_key, pop_key = jax.random.split(key, num=3)

        def _uniform_init(key, search_space):
            return jax.random.uniform(
                key,
                (pop_size,),
                minval=search_space.low,
                maxval=search_space.high,
            )

        search_space = self.config.search_space
        pop = PyTreeDict(
            {
                hp: _uniform_init(key, search_space[hp])
                for hp, key in zip(
                    search_space.keys(), jax.random.split(pop_key, len(search_space))
                )
            }
        )
        pop = tree_device_put(pop, self.sharding)

        # save metric on GPU0
        workflow_metrics = PBTWorkflowMetric()

        workflow_keys = jax.random.split(workflow_key, pop_size)
        workflow_keys = jax.device_put(workflow_keys, self.sharding)

        pop_workflow_state = jax.jit(
            jax.vmap(self.workflow.setup, spmd_axis_name=POP_AXIS_NAME),
            in_shardings=self.sharding,
            out_shardings=self.sharding,
        )(workflow_keys)

        pop_workflow_state = jax.jit(
            jax.vmap(
                self.apply_hyperparams_to_workflow_state, spmd_axis_name=POP_AXIS_NAME
            ),
            in_shardings=self.sharding,
            out_shardings=self.sharding,
        )(pop_workflow_state, pop)

        return State(
            key=key,
            metrics=workflow_metrics,
            pop_workflow_state=pop_workflow_state,
            pop=pop,
        )

    def apply_hyperparams_to_workflow_state(
        self, workflow_state: State, hyperparams: PyTreeDict[str, chex.Numeric]
    ):
        return workflow_state.replace(hp_state=hyperparams)
