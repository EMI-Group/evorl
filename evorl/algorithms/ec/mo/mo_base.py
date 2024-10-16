import numpy as np
import wandb

import chex
import jax
import orbax.checkpoint as ocp

from evorl.distributed import tree_unpmap
from evorl.types import State
from evorl.recorders import get_1d_array_statistics
from evox.operators import non_dominated_sort

from ..ec_workflow import EvoXWorkflowWrapper


class MOECWorkflowTemplate(EvoXWorkflowWrapper):
    def setup(self, key: chex.PRNGKey) -> State:
        state = super().setup(key)
        for metric_name in self.config.obj_names:
            wandb.define_metric(f"pf_objectives.{metric_name}.val", hidden=True)

        return state

    def learn(self, state: State) -> State:
        start_iteration = tree_unpmap(state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            train_metrics = tree_unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = tree_unpmap(workflow_metrics, self.pmap_axis_name)
            self.recorder.write(workflow_metrics.to_local_dict(), iters)

            cpu_device = jax.devices("cpu")[0]
            with jax.default_device(cpu_device):
                objectives = jax.device_put(train_metrics.objectives, cpu_device)
                fitnesses = objectives * self._workflow.opt_direction
                pf_rank = non_dominated_sort(fitnesses, "scan")
                pf_objectives = train_metrics.objectives[pf_rank == 0]

            train_metrics_dict = {}
            metric_names = self.config.obj_names
            objectives = np.asarray(objectives)
            pf_objectives = np.asarray(pf_objectives)
            train_metrics_dict["objectives"] = {
                metric_names[i]: get_1d_array_statistics(
                    objectives[:, i], histogram=True
                )
                for i in range(len(metric_names))
            }

            train_metrics_dict["pf_objectives"] = {
                metric_names[i]: get_1d_array_statistics(
                    pf_objectives[:, i], histogram=True
                )
                for i in range(len(metric_names))
            }
            train_metrics_dict["num_pf"] = pf_objectives.shape[0]

            self.recorder.write(train_metrics_dict, iters)

            self.checkpoint_manager.save(
                iters,
                args=ocp.args.StandardSave(
                    tree_unpmap(state, self.pmap_axis_name),
                ),
            )
