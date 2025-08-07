import logging
import numpy as np
from omegaconf import DictConfig
from typing_extensions import Self  # pytype: disable=not-supported-yet]

import jax
import jax.numpy as jnp

from evox.algorithms import NSGA2
from evox.operators import non_dominated_sort

from evorl.types import State, Params
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators import BraxEvaluator
from evorl.agent import AgentState
from evorl.distributed import unpmap
from evorl.ec.optimizers.evox_wrapper import EvoXAlgorithmAdapter, ECState
from evorl.utils.ec_utils import ParamVectorSpec
from evorl.recorders import get_1d_array_statistics
from evorl.workflows import MultiObjectiveECWorkflowTemplate

from ..obs_utils import init_obs_preprocessor
from ..ec_agent import make_deterministic_ec_agent

logger = logging.getLogger(__name__)


class NSGA2Workflow(MultiObjectiveECWorkflowTemplate):
    @classmethod
    def name(cls):
        return "NSGA2"

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        super()._rescale_config(config)

        num_devices = jax.device_count()
        if config.random_timesteps % num_devices != 0:
            logging.warning(
                f"When enable_multi_devices=True, pop_size ({config.random_timesteps}) should be divisible by num_devices ({num_devices}),"
            )

        config.random_timesteps = (config.random_timesteps // num_devices) * num_devices

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        env = create_env(
            config.env,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        agent = make_deterministic_ec_agent(
            action_space=env.action_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            use_bias=config.agent_network.use_bias,
            normalize_obs=config.normalize_obs,
            norm_layer_type=config.agent_network.norm_layer_type,
        )

        # dummy agent_state
        agent_key = jax.random.PRNGKey(config.seed)
        agent_state = agent.init(env.obs_space, env.action_space, agent_key)
        param_vec_spec = ParamVectorSpec(agent_state.params.policy_params)

        ec_optimizer = EvoXAlgorithmAdapter(
            algorithm=NSGA2(
                lb=jnp.full((param_vec_spec.vec_size,), config.agent_network.lb),
                ub=jnp.full((param_vec_spec.vec_size,), config.agent_network.ub),
                n_objs=len(config.metric_names),
                pop_size=config.pop_size,
            ),
            param_vec_spec=param_vec_spec,
        )

        if config.explore:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        ec_evaluator = BraxEvaluator(
            env=env,
            action_fn=action_fn,
            max_episode_steps=config.env.max_episode_steps,
            discount=config.discount,
            metric_names=tuple(config.metric_names),
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
                pmap_axis_name=self.pmap_axis_name,
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
        start_iteration = unpmap(state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            train_metrics = unpmap(train_metrics, self.pmap_axis_name)
            workflow_metrics = unpmap(workflow_metrics, self.pmap_axis_name)
            self.recorder.write(workflow_metrics.to_local_dict(), iters)

            cpu_device = jax.devices("cpu")[0]
            with jax.default_device(cpu_device):
                objectives = jax.device_put(train_metrics.objectives, cpu_device)
                pf_rank = non_dominated_sort(-objectives, "scan")
                pf_objectives = train_metrics.objectives[pf_rank == 0]

            train_metrics_dict = {}
            metric_names = self.config.metric_names
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
                unpmap(state, self.pmap_axis_name),
                force=i == self.config.num_iters,
            )
