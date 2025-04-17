from evorl.distributed import unpmap
from evorl.types import State
from evorl.recorders import add_prefix

from evorl.algorithms.offpolicy_utils import skip_replay_buffer_state
from evorl.algorithms.td3 import TD3Workflow


class TD3WorkflowMod(TD3Workflow):
    """TD3Workflow with total_episode termination condition."""

    def learn(self, state: State) -> State:
        sampled_episodes = unpmap(state.metrics.sampled_episodes).tolist()

        while sampled_episodes < self.config.total_episodes:
            train_metrics, state = self._multi_steps(state)
            workflow_metrics = state.metrics

            # current iteration
            iterations = unpmap(state.metrics.iterations, self.pmap_axis_name).tolist()
            train_metrics = unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = unpmap(workflow_metrics, self.pmap_axis_name)
            self.recorder.write(train_metrics.to_local_dict(), iterations)
            self.recorder.write(workflow_metrics.to_local_dict(), iterations)

            sampled_episodes = unpmap(state.metrics.sampled_episodes).tolist()

            if (
                iterations % self.config.eval_interval == 0
                or sampled_episodes >= self.config.total_episodes
            ):
                eval_metrics, state = self.evaluate(state)
                eval_metrics = unpmap(eval_metrics, self.pmap_axis_name)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iterations
                )

            saved_state = unpmap(state, self.pmap_axis_name)
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)
            self.checkpoint_manager.save(iterations, saved_state)
        self.checkpoint_manager.save(iterations, saved_state, force=True)

        return state
