import numpy as np
from omegaconf import DictConfig
from typing import Self

import jax

from evorl.types import State, Params
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators import Evaluator
from evorl.agent import AgentState
from evorl.ec.optimizers import NSGA2, ECState
from evorl.ec.operators.selection.non_dominated import non_dominated_sort
from evorl.utils.ec_utils import ParamVectorSpec
from evorl.recorders import get_1d_array_statistics
from evorl.workflows import MultiObjectiveECWorkflowTemplate

from ..obs_utils import init_obs_preprocessor
from ..ec_agent import make_deterministic_ec_agent


class NSGA2Workflow(MultiObjectiveECWorkflowTemplate):
    """Optimize a continuous-action policy using single-agent vector rewards.

    Environments return rewards of shape `(num_envs, num_objectives)` with at
    least two objectives. The objective count is inferred without executing a
    reset. `env.reward_keys` is optional and provides metric names when an
    environment uses it to select reward components. Every objective is
    maximized.
    """

    @classmethod
    def name(cls):
        return "NSGA2"

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        agent = make_deterministic_ec_agent(
            action_space=env.action_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            use_bias=config.agent_network.use_bias,
            normalize_obs=config.normalize_obs,
            norm_layer_type=config.agent_network.norm_layer_type,
            policy_obs_key=config.agent_network.policy_obs_key,
        )

        # dummy agent_state
        agent_key = jax.random.key(config.seed)
        reward = jax.eval_shape(env.reset, agent_key).reward
        if len(reward.shape) != 2 or reward.shape[1] < 2:
            raise ValueError(
                "NSGA2 requires vector rewards with shape (num_envs, num_objectives) "
                "and at least two objectives."
            )
        agent_state = agent.init(env.obs_space, env.action_space, agent_key)
        param_vec_spec = ParamVectorSpec(agent_state.params.policy_params)

        ec_optimizer = NSGA2(
            lb=config.agent_network.lb,
            ub=config.agent_network.ub,
            num_objectives=reward.shape[1],
            pop_size=config.pop_size,
            param_vec_spec=param_vec_spec,
        )

        if config.explore:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        ec_evaluator = Evaluator(
            env=env,
            action_fn=action_fn,
            max_episode_steps=config.env.max_episode_steps,
            discount=config.discount,
        )

        agent_state_vmap_axes = AgentState(
            params=0,
            obs_preprocessor_state=None,
        )

        return cls(
            config=config,
            env=env,
            agent=agent,
            ec_optimizer=ec_optimizer,
            ec_evaluator=ec_evaluator,
            agent_state_vmap_axes=agent_state_vmap_axes,
        )

    def _setup_agent_and_optimizer(self, key: jax.Array) -> tuple[AgentState, ECState]:
        agent_key, ec_key = jax.random.split(key)
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )

        ec_opt_state = self.ec_optimizer.init(ec_key)

        # remove params
        agent_state = self._replace_actor_params(agent_state, params=None)

        return agent_state, ec_opt_state

    def _postsetup(self, state: State) -> State:
        # setup obs_preprocessor_state
        if self.config.normalize_obs:
            key, obs_key = jax.random.split(state.key, 2)
            agent_state = init_obs_preprocessor(
                agent_state=state.agent_state,
                config=self.config,
                key=obs_key,
                dp_axis_name=self.dp_axis_name,
            )

            # Note: we don't count these random timesteps in state.metrics
            return state.replace(
                agent_state=agent_state,
                key=key,
            )
        else:
            return state

    def _replace_actor_params(
        self, agent_state: AgentState, params: Params
    ) -> AgentState:
        return agent_state.replace(
            params=agent_state.params.replace(policy_params=params)
        )

    def learn(self, state: State) -> State:
        """Run the remaining iterations and return the final workflow state."""
        start_iteration = state.metrics.iterations

        for i in range(start_iteration, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            self.recorder.write(workflow_metrics.to_local_dict(), iters)

            objectives = np.asarray(train_metrics.objectives)
            cpu_device = jax.devices("cpu")[0]
            with (
                jax.set_mesh(jax.sharding.Mesh(np.asarray(cpu_device), ())),
                jax.default_device(cpu_device),
            ):
                pf_rank = np.asarray(
                    non_dominated_sort(jax.device_put(objectives, cpu_device))
                )
            pf_objectives = objectives[pf_rank == 0]

            train_metrics_dict = {}
            metric_names = self.config.env.get("reward_keys") or [
                f"objective_{i}" for i in range(objectives.shape[-1])
            ]
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
                state,
                force=iters == self.config.num_iters,
            )

        return state
