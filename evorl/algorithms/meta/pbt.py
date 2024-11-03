from functools import partial

import chex
import jax
import optax
from optax.schedules import InjectStatefulHyperparamsState


from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import tree_get, tree_set

from .pbt_base import PBTWorkflowBase
from .pbt_operations import explore, select
from .utils import deepcopy_opt_state


class PBTWorkflowTemplate(PBTWorkflowBase):
    """
    Standard PBT Workflow Template
    """

    def exploit_and_explore(
        self,
        pop: chex.ArrayTree,
        pop_workflow_state: State,
        pop_metrics: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.ArrayTree, State]:
        exploit_key, explore_key = jax.random.split(key)

        config = self.config

        tops_indices, bottoms_indices = select(
            pop_metrics,  # using episode_return
            exploit_key,
            bottoms_num=round(config.pop_size * config.bottom_ratio),
            tops_num=round(config.pop_size * config.top_ratio),
        )

        parents = tree_get(pop, tops_indices)
        parents_wf_state = tree_get(pop_workflow_state, tops_indices)

        # TODO: check sharding issue with vmap under multi-devices.
        offsprings = jax.vmap(
            partial(
                explore,
                perturb_factor=config.perturb_factor,
                search_space=config.search_space,
            )
        )(parents, jax.random.split(explore_key, bottoms_indices.shape[0]))

        # Note: no need to deepcopy parents_wf_state here, since it should be
        # ensured immutable in apply_hyperparams_to_workflow_state()
        offsprings_workflow_state = jax.vmap(self.apply_hyperparams_to_workflow_state)(
            parents_wf_state, offsprings
        )

        # ==== survival | merge population ====
        pop = tree_set(pop, offsprings, bottoms_indices, unique_indices=True)
        # we copy wf_state back to offspring wf_state
        pop_workflow_state = tree_set(
            pop_workflow_state,
            offsprings_workflow_state,
            bottoms_indices,
            unique_indices=True,
        )

        return pop, pop_workflow_state


class PBTWorkflow(PBTWorkflowTemplate):
    """
    A minimal Example of PBT that tunes the lr of PPO.
    """

    @classmethod
    def name(cls):
        return "PBT"

    def _customize_optimizer(self) -> None:
        """
        Customize the target workflow's optimizer
        """
        self.workflow.optimizer = optax.inject_hyperparams(
            optax.adam, static_args=("b1", "b2", "eps", "eps_root")
        )(learning_rate=self.config.search_space.lr.low)

    def _setup_pop(self, key: chex.PRNGKey) -> chex.ArrayTree:
        pop = PyTreeDict(
            lr=jax.random.uniform(
                key,
                (self.config.pop_size,),
                minval=self.config.search_space.lr.low,
                maxval=self.config.search_space.lr.high,
            )
        )

        return pop

    def apply_hyperparams_to_workflow_state(
        self, workflow_state: State, hyperparams: PyTreeDict[str, chex.Numeric]
    ) -> State:
        """
        Note1: InjectStatefulHyperparamsState is NamedTuple, which is not immutable.
        Note2: try to avoid deepcopy unnessary state
        """
        opt_state = workflow_state.opt_state
        assert isinstance(opt_state, InjectStatefulHyperparamsState)

        opt_state = deepcopy_opt_state(opt_state)
        opt_state.hyperparams["learning_rate"] = hyperparams.lr
        return workflow_state.replace(opt_state=opt_state)
