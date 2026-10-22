from flax import nnx
from omegaconf import DictConfig
from typing import Self

import jax
import jax.tree_util as jtu

from evorl.types import State, Params
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators import Evaluator, EpisodeObsCollector
from evorl.sample_batch import SampleBatch
from evorl.agent import AgentState
from evorl.ec.optimizers import ARS, ECState
from evorl.utils import running_statistics

from .es_workflow import ESWorkflowTemplate
from ..obs_utils import init_obs_preprocessor
from ..ec_agent import DeterministicECAgent


class ARSWorkflow(ESWorkflowTemplate):
    @classmethod
    def name(cls):
        return "ARS"

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        agent = DeterministicECAgent(
            agent_network_cfg=config.agent_network,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
        )

        ec_optimizer = ARS(
            pop_size=config.pop_size,
            num_elites=config.num_elites,
            lr=config.lr,
            noise_std=config.noise_std,
            optimizer_name=config.optimizer_name,
        )

        if config.explore:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        assert config.normalize_obs_mode in ["VBN", "RS", "Global"]
        if config.normalize_obs_mode == "VBN":
            ec_evaluator = Evaluator(
                env=env,
                action_fn=action_fn,
                max_episode_steps=config.env.max_episode_steps,
                discount=config.discount,
            )
        else:
            ec_evaluator = EpisodeObsCollector(
                env=env,
                action_fn=action_fn,
                max_episode_steps=config.env.max_episode_steps,
                discount=config.discount,
            )

        # to evaluate the pop-mean actor
        eval_env = create_env(
            config.env,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        agent_state_vmap_axes = AgentState(
            model_state=nnx.StateAxes({nnx.Param: 0, ...: None}),
            obs_preprocessor_state=None,
        )

        return cls(
            config=config,
            env=env,
            agent=agent,
            ec_optimizer=ec_optimizer,
            ec_evaluator=ec_evaluator,
            evaluator=evaluator,
            agent_state_vmap_axes=agent_state_vmap_axes,
        )

    def _setup_agent_and_optimizer(self, key: jax.Array) -> tuple[AgentState, ECState]:
        agent_key, ec_key = jax.random.split(key)
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )

        init_actor_params = nnx.state(
            nnx.merge(agent_state.model_state, copy=True).policy, nnx.Param
        )
        ec_opt_state = self.ec_optimizer.init(init_actor_params, ec_key)

        # remove params
        agent_state = self._replace_actor_params(agent_state, params=None)

        return agent_state, ec_opt_state

    def _postsetup(self, state: State) -> State:
        # setup obs_preprocessor_state
        if self.config.normalize_obs and self.config.normalize_obs_mode != "RS":
            key, obs_key = jax.random.split(state.key, 2)
            agent_state = init_obs_preprocessor(
                agent_state=state.agent_state,
                config=self.config,
                key=obs_key,
                axis_name=self.dp_axis_name,
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
        models = nnx.merge(agent_state.model_state, copy=True)
        if params is None:
            params = jax.tree.map(lambda _: None, nnx.state(models.policy, nnx.Param))
        nnx.update(models.policy, params)
        return agent_state.replace(model_state=nnx.unpack(models))

    def _get_pop_center(self, state: State) -> AgentState:
        pop_center = state.ec_opt_state.mean

        return self._replace_actor_params(state.agent_state, pop_center)

    def _update_obs_preprocessor(
        self, agent_state: AgentState, trajectory: SampleBatch
    ) -> AgentState:
        if self.config.normalize_obs_mode == "Global":
            obs_preprocessor_state = running_statistics.update(
                agent_state.obs_preprocessor_state,
                trajectory.obs,
                weights=1 - trajectory.dones,
                axis_name=self.dp_axis_name,
            )

        elif self.config.normalize_obs_mode == "RS":
            dummy_obs = jtu.tree_map(lambda x: x[0, 0], trajectory.obs)
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
            obs_preprocessor_state = running_statistics.update(
                obs_preprocessor_state,
                trajectory.obs,
                weights=1 - trajectory.dones,
                axis_name=self.dp_axis_name,
            )
        else:
            obs_preprocessor_state = agent_state.obs_preprocessor_state

        return agent_state.replace(obs_preprocessor_state=obs_preprocessor_state)
