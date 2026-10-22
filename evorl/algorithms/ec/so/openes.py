from evorl.utils import running_statistics
from flax import nnx
from omegaconf import DictConfig
from typing import Self

import jax

from evorl.types import State, Params
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators import Evaluator
from evorl.agent import AgentState
from evorl.ec.optimizers import OpenES, ExponentialScheduleSpec, ECState


from .es_workflow import ESWorkflowTemplate
from ..obs_utils import init_obs_preprocessor
from ..ec_agent import DeterministicECAgent


class OpenESWorkflow(ESWorkflowTemplate):
    @classmethod
    def name(cls):
        return "OpenES"

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

        ec_optimizer = OpenES(
            pop_size=config.pop_size,
            lr_schedule=ExponentialScheduleSpec(**config.ec_lr),
            noise_std_schedule=ExponentialScheduleSpec(**config.ec_noise_std),
            mirror_sampling=config.mirror_sampling,
            weight_decay=config.weight_decay,
            optimizer_name=config.optimizer_name,
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
        if self.config.normalize_obs:
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
